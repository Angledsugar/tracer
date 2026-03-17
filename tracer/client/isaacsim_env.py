"""
Isaac Sim Environment Wrapper
Isaac Sim과 DVA 파이프라인을 연결하는 인터페이스
"""

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class IsaacSimEnv:
    """
    Isaac Sim 환경 래퍼.

    Isaac Sim에서 로봇 시뮬레이션을 실행하고,
    DVA 파이프라인에 필요한 관찰/행동 인터페이스를 제공합니다.

    NOTE: 실제 Isaac Sim API 호출은 Isaac Sim이 설치된 환경에서
    omni.isaac 모듈을 import하여 사용합니다.
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

        self._sim = None
        self._robot = None
        self._camera = None
        self._initialized = False

    def initialize(self):
        """Isaac Sim 환경 초기화."""
        logger.info(f"Initializing Isaac Sim: task={self.task_name}, robot={self.robot_type}")

        try:
            # Isaac Sim imports - only available in Isaac Sim environment
            from isaacsim import SimulationApp
            simulation_app = SimulationApp({"headless": self.headless})

            import omni.isaac.core as isaac_core
            from omni.isaac.core import World
            from omni.isaac.core.robots import Robot

            self._sim = World(stage_units_in_meters=1.0)

            # TODO: Load specific task scene and robot
            # This depends on the task configuration
            # self._setup_scene()
            # self._setup_robot()
            # self._setup_camera()

            self._initialized = True
            logger.info("Isaac Sim initialized successfully")

        except ImportError:
            logger.warning(
                "Isaac Sim not available - using placeholder environment. "
                "Install Isaac Sim to use the real simulator."
            )
            self._init_placeholder()

    def _init_placeholder(self):
        """Isaac Sim 없이 개발/테스트용 플레이스홀더."""
        self._initialized = True
        self._ee_pos = np.array([0.4, 0.0, 0.3])    # end-effector position
        self._ee_rot = np.array([0.0, 0.0, 0.0])     # end-effector rotation (euler)
        self._gripper_state = 0.0                      # 0: open, 1: closed
        logger.info("Placeholder environment initialized")

    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        """
        환경 리셋.

        Returns:
            (frame, proprioception)
        """
        if not self._initialized:
            self.initialize()

        if self._sim:
            self._sim.reset()

        self._ee_pos = np.array([0.4, 0.0, 0.3])
        self._ee_rot = np.array([0.0, 0.0, 0.0])
        self._gripper_state = 0.0

        return self.get_observation()

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
        if self._camera:
            # Isaac Sim camera
            # return self._camera.get_rgba()[:, :, :3]
            pass

        # Placeholder: 간단한 시각화 이미지 생성
        h, w = self.camera_resolution
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # 배경
        frame[:, :] = [40, 40, 60]

        # 테이블 (간단한 사각형)
        frame[h//2:, w//4:3*w//4] = [139, 90, 43]

        # End-effector 위치를 이미지에 표시
        px = int((self._ee_pos[0] - 0.2) / 0.4 * w)
        py = int((1.0 - (self._ee_pos[2] - 0.0) / 0.6) * h)
        px = np.clip(px, 5, w-5)
        py = np.clip(py, 5, h-5)

        # 그리퍼 표시
        color = [255, 0, 0] if self._gripper_state > 0.5 else [0, 255, 0]
        frame[py-4:py+4, px-4:px+4] = color

        return frame

    def _get_proprioception(self) -> np.ndarray:
        """로봇 고유수용감각 반환."""
        return np.concatenate([
            self._ee_pos,
            self._ee_rot,
            [self._gripper_state],
        ])

    def execute_action(self, action: np.ndarray):
        """
        로봇 행동 실행.

        Args:
            action: [7] (dx, dy, dz, drx, dry, drz, gripper)
        """
        if self._sim:
            # Isaac Sim robot control
            # self._robot.apply_action(action)
            # self._sim.step()
            pass

        # Placeholder: 직접 상태 업데이트
        self._ee_pos += action[:3] * 0.01   # scale down for safety
        self._ee_rot += action[3:6] * 0.01
        self._gripper_state = float(action[6] > 0.5)

        # 작업 공간 제한
        self._ee_pos = np.clip(self._ee_pos, [0.1, -0.3, 0.0], [0.7, 0.3, 0.6])

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, bool, dict]:
        """
        Gym-style step interface (for RL training).

        Returns:
            frame, proprioception, reward, done, info
        """
        self.execute_action(action)

        if self._sim:
            self._sim.step()

        frame, proprio = self.get_observation()

        # TODO: 작업별 보상 함수 정의
        reward = 0.0
        done = False
        info = {"step": 0}

        return frame, proprio, reward, done, info

    def close(self):
        """환경 종료."""
        if self._sim:
            self._sim.stop()
        logger.info("Environment closed")
