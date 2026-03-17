"""
Isaac Sim Environment Wrapper
Isaac Sim과 DVA 파이프라인을 연결하는 인터페이스

Isaac Sim Core API (4.5/5.x)를 사용하여
Franka 로봇 + 카메라 + Pick-and-Place 씬을 구성합니다.
Isaac Sim이 설치되지 않은 환경에서는 placeholder로 동작합니다.
"""

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# 큐브 초기 위치 (테이블 위)
CUBE_SPAWN_POS = np.array([0.4, 0.1, 0.55])
# 바닥 높이 임계값 (이 아래로 내려가면 리스폰)
GROUND_THRESHOLD = 0.03
# Franka 홈 포지션 (팔을 접은 상태, 테이블 충돌 방지)
# 7 arm joints + 2 finger joints
FRANKA_HOME_JOINTS = np.array([
    0.0,       # panda_joint1
    -0.785,    # panda_joint2 (-π/4, 어깨 뒤로)
    0.0,       # panda_joint3
    -2.356,    # panda_joint4 (-3π/4, 팔꿈치 접기)
    0.0,       # panda_joint5
    1.571,     # panda_joint6 (π/2, 손목 올리기)
    0.785,     # panda_joint7 (π/4)
    0.04,      # panda_finger_joint1 (열림)
    0.04,      # panda_finger_joint2 (열림)
])


class IsaacSimEnv:
    """
    Isaac Sim 환경 래퍼.

    Isaac Sim에서 로봇 시뮬레이션을 실행하고,
    DVA 파이프라인에 필요한 관찰/행동 인터페이스를 제공합니다.
    """

    def __init__(
        self,
        task_name: str = "FrankaPickAndPlace",
        robot_type: str = "franka",
        camera_resolution: tuple[int, int] = (256, 256),
        control_mode: str = "end_effector",
        headless: bool = False,
    ):
        self.task_name = task_name
        self.robot_type = robot_type
        self.camera_resolution = camera_resolution
        self.control_mode = control_mode
        self.headless = headless

        self._sim_app = None
        self._world = None
        self._robot = None
        self._camera = None
        self._target_cube = None
        self._initialized = False
        self._use_placeholder = False

        # UI 표시용
        self._prediction_window = None
        self._prediction_provider = None
        self._camera_window = None
        self._camera_provider = None
        self._prompt_window = None
        self._current_prompt = ""

        # Placeholder state
        self._ee_pos = np.array([0.4, 0.0, 0.3])
        self._ee_rot = np.array([0.0, 0.0, 0.0])
        self._gripper_state = 0.0

    def initialize(self):
        """Isaac Sim 환경 초기화."""
        logger.info(f"Initializing Isaac Sim: task={self.task_name}, robot={self.robot_type}")

        try:
            self._init_isaac_sim()
        except ImportError as e:
            logger.warning(
                f"Isaac Sim not available ({e}) - using placeholder environment. "
                "Install Isaac Sim to use the real simulator."
            )
            self._init_placeholder()

    def _init_isaac_sim(self):
        """실제 Isaac Sim 환경 초기화."""
        # SimulationApp은 반드시 다른 omniverse import 전에 생성해야 함
        from isaacsim import SimulationApp

        self._sim_app = SimulationApp({
            "headless": self.headless,
            "width": 1280,
            "height": 720,
            "anti_aliasing": 0,
            "renderer": "RasterOnly",  # RTX 레이트레이싱 비활성화 (VRAM 절약)
            "max_gpu_count": 1,
        })

        # SimulationApp 생성 후에만 omniverse 모듈 import 가능
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
        from isaacsim.robot.manipulators.examples.franka import Franka
        from isaacsim.sensors.camera import Camera
        import isaacsim.core.utils.numpy.rotations as rot_utils

        # World 생성
        self._world = World(stage_units_in_meters=1.0)
        self._world.scene.add_default_ground_plane()

        # 조명 추가 (RasterOnly 렌더러에서는 조명이 없으면 검은 화면)
        import omni.kit.commands
        from pxr import UsdLux, Sdf

        stage = self._world.stage

        # Dome light (전체 환경 조명)
        dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
        dome_light.GetIntensityAttr().Set(1000.0)

        # Distant light (방향 조명 - 태양광 역할)
        distant_light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DistantLight"))
        distant_light.GetIntensityAttr().Set(3000.0)
        distant_light.GetAngleAttr().Set(0.53)

        # Franka 로봇 추가
        self._robot = self._world.scene.add(
            Franka(prim_path="/World/Franka", name="franka")
        )

        # 테이블 추가
        self._world.scene.add(
            FixedCuboid(
                prim_path="/World/Table",
                name="table",
                position=np.array([0.4, 0.0, 0.25]),
                scale=np.array([0.6, 0.8, 0.5]),
                color=np.array([0.5, 0.3, 0.15]),
            )
        )

        # 타겟 큐브 추가 (pick-and-place 대상)
        self._target_cube = self._world.scene.add(
            DynamicCuboid(
                prim_path="/World/TargetCube",
                name="target_cube",
                position=CUBE_SPAWN_POS.copy(),
                scale=np.array([0.04, 0.04, 0.04]),
                color=np.array([1.0, 0.0, 0.0]),
            )
        )

        # 카메라 설정: 로봇 베이스 기준 높이 2m, 45도 하향 뷰 (책상 전체 보임)
        h, w = self.camera_resolution
        self._camera = Camera(
            prim_path="/World/Camera",
            position=np.array([1.9, 0.0, 2.0]),
            frequency=20,
            resolution=(w, h),
            orientation=rot_utils.euler_angles_to_quats(
                np.array([0, 45, 180]), degrees=True
            ),
        )

        # 초기화
        self._world.reset()
        self._camera.initialize()

        # 로봇을 홈 포지션으로 설정 (팔 접기 → 테이블 충돌 방지)
        self._robot.set_joint_positions(FRANKA_HOME_JOINTS)

        # 물리 안정화 + 카메라 렌더링 준비
        for _ in range(30):
            self._world.step(render=True)

        # UI 설정 (headless가 아닌 경우)
        if not self.headless:
            self._init_prediction_display()
            self._init_camera_viewport()
            self._init_prompt_input()

        self._initialized = True
        self._use_placeholder = False
        logger.info("Isaac Sim initialized successfully")

    def _init_camera_viewport(self):
        """메인 viewport의 활성 카메라를 /World/Camera로 설정."""
        try:
            from omni.kit.viewport.utility import get_active_viewport
            viewport = get_active_viewport()
            if viewport:
                viewport.set_active_camera("/World/Camera")
                logger.info("Viewport active camera set to /World/Camera")
            else:
                logger.warning("No active viewport found")
        except Exception as e:
            logger.warning(f"Could not set viewport camera: {e}")

    def _init_prompt_input(self):
        """실시간 프롬프트 입력 UI 창 생성."""
        try:
            import omni.ui as ui

            self._prompt_window = ui.Window(
                "Language Prompt", width=400, height=120,
            )
            with self._prompt_window.frame:
                with ui.VStack(spacing=5):
                    ui.Label("Language Instruction (Enter to apply)", height=20)
                    field = ui.StringField(height=30)
                    field.model.set_value(self._current_prompt)
                    self._prompt_label = ui.Label(
                        f"Active: {self._current_prompt}", height=20,
                        style={"color": 0xFF00FF00},
                    )

                    def _on_prompt_changed(model):
                        new_prompt = model.get_value_as_string()
                        self._current_prompt = new_prompt
                        self._prompt_label.text = f"Active: {new_prompt}"
                        logger.info(f"Prompt updated: {new_prompt}")

                    field.model.add_end_edit_fn(_on_prompt_changed)

            logger.info("Prompt input window created")
        except Exception as e:
            logger.warning(f"Could not create prompt input: {e}")

    def set_initial_prompt(self, prompt: str):
        """초기 프롬프트 설정."""
        self._current_prompt = prompt

    def get_current_prompt(self) -> str:
        """현재 활성 프롬프트 반환. UI에서 변경된 값을 반환."""
        return self._current_prompt

    def _init_prediction_display(self):
        """카메라 입력 + Cosmos 예측 프레임을 표시할 omni.ui 창 생성."""
        try:
            import omni.ui as ui

            h, w = self.camera_resolution
            blank = [0] * (w * h * 4)

            # 카메라 입력 이미지 창
            self._camera_window = ui.Window(
                "Camera Input", width=w + 20, height=h + 40,
            )
            with self._camera_window.frame:
                with ui.VStack():
                    ui.Label("Isaac Sim Camera (Context Frame)", height=20)
                    self._camera_provider = ui.ByteImageProvider()
                    self._camera_provider.set_bytes_data(blank, [w, h])
                    ui.ImageWithProvider(
                        self._camera_provider, width=w, height=h,
                    )

            # Cosmos 예측 이미지 창
            self._prediction_window = ui.Window(
                "Cosmos Prediction", width=w + 20, height=h + 40,
            )
            with self._prediction_window.frame:
                with ui.VStack():
                    ui.Label("Cosmos Predicted Frame", height=20)
                    self._prediction_provider = ui.ByteImageProvider()
                    self._prediction_provider.set_bytes_data(blank, [w, h])
                    ui.ImageWithProvider(
                        self._prediction_provider, width=w, height=h,
                    )

            logger.info("Display windows created (Camera Input + Cosmos Prediction)")
        except Exception as e:
            logger.warning(f"Could not create display windows: {e}")
            self._prediction_window = None
            self._prediction_provider = None
            self._camera_window = None
            self._camera_provider = None

    def _frame_to_rgba_list(self, frame: np.ndarray) -> tuple[list, int, int]:
        """RGB numpy 프레임을 omni.ui ByteImageProvider용 RGBA 리스트로 변환."""
        h, w = frame.shape[:2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = frame
        rgba[:, :, 3] = 255
        return rgba.flatten().tolist(), w, h

    def _update_camera_display(self, frame: np.ndarray):
        """카메라 입력 이미지를 UI 창에 표시."""
        if self._camera_provider is None:
            return
        try:
            data, w, h = self._frame_to_rgba_list(frame)
            self._camera_provider.set_bytes_data(data, [w, h])
        except Exception as e:
            logger.debug(f"Failed to update camera display: {e}")

    def show_predicted_frames(self, frames: list[np.ndarray], frame_index: int = 0):
        """
        Cosmos가 예측한 프레임을 Isaac Sim UI 창에 표시.

        Args:
            frames: Cosmos가 생성한 미래 프레임 리스트 [H, W, 3] uint8
            frame_index: 표시할 프레임 인덱스
        """
        if self._prediction_provider is None or not frames:
            return

        try:
            frame = frames[min(frame_index, len(frames) - 1)]
            data, w, h = self._frame_to_rgba_list(frame)
            self._prediction_provider.set_bytes_data(data, [w, h])
        except Exception as e:
            logger.debug(f"Failed to update prediction display: {e}")

    def _init_placeholder(self):
        """Isaac Sim 없이 개발/테스트용 플레이스홀더."""
        self._initialized = True
        self._use_placeholder = True
        self._ee_pos = np.array([0.4, 0.0, 0.3])
        self._ee_rot = np.array([0.0, 0.0, 0.0])
        self._gripper_state = 0.0
        logger.info("Placeholder environment initialized")

    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        """
        환경 리셋.

        Returns:
            (frame, proprioception)
        """
        if not self._initialized:
            self.initialize()

        if not self._use_placeholder:
            self._world.reset()
            # 로봇을 홈 포지션으로 리셋
            self._robot.set_joint_positions(FRANKA_HOME_JOINTS)
            # 큐브 위치 리셋
            self._respawn_cube()
            # 물리 안정화
            for _ in range(20):
                self._world.step(render=False)
        else:
            self._ee_pos = np.array([0.4, 0.0, 0.3])
            self._ee_rot = np.array([0.0, 0.0, 0.0])
            self._gripper_state = 0.0

        return self.get_observation()

    def _respawn_cube(self):
        """큐브를 테이블 위 초기 위치로 리스폰."""
        if self._target_cube is not None:
            self._target_cube.set_world_pose(
                position=CUBE_SPAWN_POS.copy(),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            )
            self._target_cube.set_linear_velocity(np.zeros(3))
            self._target_cube.set_angular_velocity(np.zeros(3))
            logger.info("Cube respawned on table")

    def _check_cube_fallen(self):
        """큐브가 바닥에 닿았는지 확인하고, 닿았으면 리스폰."""
        if self._use_placeholder or self._target_cube is None:
            return

        cube_pos, _ = self._target_cube.get_world_pose()
        if cube_pos[2] < GROUND_THRESHOLD:
            logger.info(
                f"Cube fell to ground (z={cube_pos[2]:.3f}), respawning..."
            )
            self._respawn_cube()

    def get_observation(self) -> tuple[np.ndarray, np.ndarray]:
        """
        현재 관찰 반환.

        Returns:
            frame: [H, W, 3] uint8 RGB image
            proprioception: [7] float (x, y, z, rx, ry, rz, gripper)
        """
        frame = self._get_camera_image()
        proprio = self._get_proprioception()
        return frame, proprio

    def _get_camera_image(self) -> np.ndarray:
        """카메라 이미지 캡처."""
        if not self._use_placeholder:
            # Isaac Sim 카메라에서 RGBA 이미지를 가져와 RGB로 변환
            rgba = self._camera.get_rgba()

            # 카메라가 아직 렌더링되지 않은 경우 빈 프레임 반환
            h, w = self.camera_resolution
            if rgba is None or rgba.ndim != 3:
                logger.warning("Camera not ready, returning blank frame")
                return np.zeros((h, w, 3), dtype=np.uint8)

            # float (0~1) → uint8 (0~255) 변환
            if rgba.dtype in (np.float32, np.float64):
                rgb = (rgba[:, :, :3] * 255).clip(0, 255).astype(np.uint8)
            else:
                rgb = rgba[:, :, :3].astype(np.uint8)

            # 카메라 입력 창에 표시
            self._update_camera_display(rgb)

            return rgb

        # Placeholder: 간단한 시각화 이미지 생성
        h, w = self.camera_resolution
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # 배경
        frame[:, :] = [40, 40, 60]

        # 테이블 (갈색 사각형)
        frame[h // 2:, w // 4:3 * w // 4] = [139, 90, 43]

        # End-effector 위치를 이미지에 표시
        px = int((self._ee_pos[0] - 0.2) / 0.4 * w)
        py = int((1.0 - (self._ee_pos[2] - 0.0) / 0.6) * h)
        px = np.clip(px, 5, w - 5)
        py = np.clip(py, 5, h - 5)

        # 그리퍼 색상: 빨간색(닫힘) / 초록색(열림)
        color = [255, 0, 0] if self._gripper_state > 0.5 else [0, 255, 0]
        frame[py - 4:py + 4, px - 4:px + 4] = color

        # 타겟 큐브 (빨간 블록) 표시
        cube_px = int((0.4 - 0.2) / 0.4 * w)
        cube_py = int((1.0 - (0.55 - 0.0) / 0.6) * h)
        cube_px = np.clip(cube_px, 8, w - 8)
        cube_py = np.clip(cube_py, 8, h - 8)
        frame[cube_py - 6:cube_py + 6, cube_px - 6:cube_px + 6] = [200, 30, 30]

        return frame

    def _get_proprioception(self) -> np.ndarray:
        """로봇 고유수용감각 (end-effector pose + gripper state) 반환."""
        if not self._use_placeholder:
            # End-effector 위치/자세
            ee_pos, ee_quat = self._robot.end_effector.get_world_pose()

            # Quaternion → Euler angles (roll, pitch, yaw)
            import isaacsim.core.utils.numpy.rotations as rot_utils
            ee_euler = rot_utils.quats_to_euler_angles(ee_quat.reshape(1, 4))[0]

            # Gripper 상태: 열린 정도를 0~1로 정규화
            gripper_pos = self._robot.gripper.get_joint_positions()
            open_pos = self._robot.gripper.joint_opened_positions
            gripper_normalized = float(np.mean(gripper_pos) / np.mean(open_pos))
            # 0 = 닫힘, 1 = 열림 → 반전하여 1 = 닫힘으로 통일
            gripper_state = 1.0 - np.clip(gripper_normalized, 0.0, 1.0)

            return np.concatenate([ee_pos, ee_euler, [gripper_state]])

        return np.concatenate([
            self._ee_pos,
            self._ee_rot,
            [self._gripper_state],
        ])

    def execute_action(self, action: np.ndarray):
        """
        End-effector delta 명령으로 로봇 행동 실행.

        Args:
            action: [7] (dx, dy, dz, drx, dry, drz, gripper)
                    - dx/dy/dz: end-effector 위치 변화량
                    - drx/dry/drz: end-effector 회전 변화량 (euler)
                    - gripper: > 0.5이면 닫힘, ≤ 0.5이면 열림
        """
        if not self._use_placeholder:
            self._execute_action_isaac(action)
            # 매 액션 실행 후 큐브가 떨어졌는지 확인
            self._check_cube_fallen()
            return

        # Placeholder: 직접 상태 업데이트
        self._ee_pos += action[:3] * 0.01
        self._ee_rot += action[3:6] * 0.01
        self._gripper_state = float(action[6] > 0.5)

        # 작업 공간 제한
        self._ee_pos = np.clip(self._ee_pos, [0.1, -0.3, 0.0], [0.7, 0.3, 0.6])

    def _execute_action_isaac(self, action: np.ndarray):
        """Isaac Sim에서 delta end-effector 명령 실행."""
        from isaacsim.robot.manipulators.examples.franka.controllers import (
            RMPFlowController,
        )
        import isaacsim.core.utils.numpy.rotations as rot_utils

        # 현재 end-effector 상태
        ee_pos, ee_quat = self._robot.end_effector.get_world_pose()

        # Delta를 적용한 타겟 위치/자세 계산
        target_pos = ee_pos + action[:3]

        # Delta rotation → 새 quaternion
        current_euler = rot_utils.quats_to_euler_angles(ee_quat.reshape(1, 4))[0]
        target_euler = current_euler + action[3:6]
        target_quat = rot_utils.euler_angles_to_quats(target_euler.reshape(1, 3))[0]

        # 작업 공간 제한
        target_pos = np.clip(
            target_pos,
            [0.1, -0.4, 0.02],
            [0.7, 0.4, 0.6],
        )

        # RMPFlow 컨트롤러로 IK 풀기
        if not hasattr(self, "_rmpflow_controller"):
            self._rmpflow_controller = RMPFlowController(
                name="rmpflow_controller",
                robot_articulation=self._robot,
            )

        joint_actions = self._rmpflow_controller.forward(
            target_end_effector_position=target_pos,
            target_end_effector_orientation=target_quat,
        )
        self._robot.apply_action(joint_actions)

        # Gripper 제어
        if action[6] > 0.5:
            self._robot.gripper.set_joint_positions(
                self._robot.gripper.joint_closed_positions
            )
        else:
            self._robot.gripper.set_joint_positions(
                self._robot.gripper.joint_opened_positions
            )

        # 시뮬레이션 스텝 진행
        self._world.step(render=not self.headless)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, bool, dict]:
        """
        Gym-style step interface.

        Returns:
            frame, proprioception, reward, done, info
        """
        self.execute_action(action)

        if not self._use_placeholder and self._world:
            self._world.step(render=not self.headless)

        frame, proprio = self.get_observation()

        reward = 0.0
        done = False
        info = {"step": 0}

        # Pick-and-place 작업: 큐브와 그리퍼 거리로 보상 계산
        if not self._use_placeholder and self._target_cube is not None:
            cube_pos, _ = self._target_cube.get_world_pose()
            ee_pos = proprio[:3]
            dist = np.linalg.norm(ee_pos - cube_pos)
            reward = -dist  # 거리가 가까울수록 높은 보상

        return frame, proprio, reward, done, info

    def close(self):
        """환경 종료."""
        if self._sim_app is not None:
            self._sim_app.close()
        logger.info("Environment closed")
