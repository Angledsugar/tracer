"""
Leapfrog Controller
추론과 실행을 겹쳐서 연속적인 로봇 제어를 수행
"""

import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np

from client.cosmos_client import CosmosClient
from models.inverse_dynamics import InverseDynamicsModel

logger = logging.getLogger(__name__)


@dataclass
class LeapfrogConfig:
    # Cosmos inference settings
    num_output_frames: int = 12
    guidance_scale: float = 7.0
    num_denoise_steps: int = 35

    # Leapfrog settings
    execute_frames: int = 8       # 12프레임 중 실행할 프레임 수
    overlap_frames: int = 4       # 다음 추론과 겹치는 프레임 수
    control_hz: float = 20.0      # 로봇 제어 주파수

    # Context settings
    max_context_frames: int = 50  # 최대 context 길이


@dataclass
class ControlState:
    """Leapfrog 제어 루프의 현재 상태."""
    current_actions: list = field(default_factory=list)
    action_index: int = 0
    previous_actions: list = field(default_factory=list)
    is_inferencing: bool = False
    pending_frames: Optional[list] = None
    pending_actions: Optional[list] = None
    step_count: int = 0


class LeapfrogController:
    """
    DVA Leapfrog Inference Controller.

    추론과 실행을 겹쳐서 연속적인 closed-loop 제어를 수행합니다.

    흐름:
    1. 현재 관찰로 Cosmos에 비디오 예측 요청 (비동기)
    2. 이전 예측의 action을 실행
    3. 새 예측이 도착하면 action 교체
    4. 반복
    """

    def __init__(
        self,
        cosmos_client: CosmosClient,
        idm: InverseDynamicsModel,
        config: LeapfrogConfig,
        get_observation: Callable[[], tuple[np.ndarray, np.ndarray]],
        execute_action: Callable[[np.ndarray], None],
        language_instruction: str = "",
        idm_device: str = "cuda:0",
    ):
        """
        Args:
            cosmos_client: Remote Cosmos server client
            idm: Inverse dynamics model (local)
            config: Leapfrog configuration
            get_observation: Function returning (frame [H,W,3], proprioception [7])
            execute_action: Function to send action to robot
            language_instruction: Task description
            idm_device: Device for IDM inference
        """
        self.cosmos = cosmos_client
        self.idm = idm
        self.config = config
        self.get_observation = get_observation
        self.execute_action = execute_action
        self.language = language_instruction
        self.idm_device = idm_device

        self.state = ControlState()
        self.context_frames: list[np.ndarray] = []
        self._running = False
        self._inference_thread: Optional[threading.Thread] = None

    def _request_inference_async(self):
        """비동기로 Cosmos 추론 요청."""
        self.state.is_inferencing = True

        context = list(self.context_frames)  # snapshot
        prev_actions = list(self.state.previous_actions)

        def _run():
            try:
                generated_frames, latency_ms = self.cosmos.predict(
                    context_frames=context,
                    language_instruction=self.language,
                    previous_actions=prev_actions,
                    num_output_frames=self.config.num_output_frames,
                    guidance_scale=self.config.guidance_scale,
                    num_denoise_steps=self.config.num_denoise_steps,
                )

                # IDM: 생성된 비디오를 행동으로 변환
                current_frame = context[-1]
                proprio = self.get_observation()[1]  # 현재 proprioception

                actions = self.idm.predict_chunk(
                    current_frame=current_frame,
                    future_frames=generated_frames,
                    proprioception=proprio,
                    device=self.idm_device,
                )

                self.state.pending_frames = generated_frames
                self.state.pending_actions = actions

                logger.info(
                    f"Inference done: {latency_ms:.0f}ms, "
                    f"{len(actions)} actions predicted"
                )

            except Exception as e:
                logger.error(f"Inference failed: {e}")
            finally:
                self.state.is_inferencing = False

        self._inference_thread = threading.Thread(target=_run, daemon=True)
        self._inference_thread.start()

    def _swap_actions(self):
        """새 추론 결과로 action 교체."""
        if self.state.pending_actions is not None:
            # 현재 실행 중인 action들을 이전 action으로 저장 (Leapfrog continuity)
            self.state.previous_actions = [
                {
                    "dx": a[0], "dy": a[1], "dz": a[2],
                    "drx": a[3], "dry": a[4], "drz": a[5],
                    "gripper": a[6],
                }
                for a in self.state.current_actions
            ]

            self.state.current_actions = self.state.pending_actions
            self.state.action_index = 0
            self.state.pending_actions = None
            self.state.pending_frames = None

            logger.debug("Actions swapped")

    def _update_context(self, frame: np.ndarray):
        """Context에 새 프레임 추가."""
        self.context_frames.append(frame)
        if len(self.context_frames) > self.config.max_context_frames:
            self.context_frames.pop(0)

    def run(self, max_steps: Optional[int] = None):
        """
        메인 Leapfrog 제어 루프.

        Args:
            max_steps: 최대 스텝 수 (None이면 무한)
        """
        self._running = True
        dt = 1.0 / self.config.control_hz

        logger.info(
            f"Leapfrog started: {self.config.control_hz}Hz, "
            f"execute={self.config.execute_frames}, "
            f"overlap={self.config.overlap_frames}"
        )

        # 초기 관찰 수집
        frame, proprio = self.get_observation()
        self._update_context(frame)

        # 첫 번째 추론 (동기 - 첫 action이 필요하므로)
        logger.info("Initial inference (synchronous)...")
        generated_frames, latency_ms = self.cosmos.predict(
            context_frames=self.context_frames,
            language_instruction=self.language,
            num_output_frames=self.config.num_output_frames,
            guidance_scale=self.config.guidance_scale,
            num_denoise_steps=self.config.num_denoise_steps,
        )
        self.state.current_actions = self.idm.predict_chunk(
            current_frame=frame,
            future_frames=generated_frames,
            proprioception=proprio,
            device=self.idm_device,
        )
        logger.info(f"Initial inference done: {latency_ms:.0f}ms")

        # 메인 루프
        while self._running:
            if max_steps and self.state.step_count >= max_steps:
                break

            loop_start = time.time()

            # 1. 현재 관찰 수집
            frame, proprio = self.get_observation()
            self._update_context(frame)

            # 2. 추론 완료 시 action 교체 (새 추론 시작 전에 먼저 체크)
            if not self.state.is_inferencing and self.state.pending_actions:
                self._swap_actions()

            # 3. 현재 action 실행
            if self.state.action_index < len(self.state.current_actions):
                action = self.state.current_actions[self.state.action_index]
                self.execute_action(action)
                self.state.action_index += 1
            else:
                logger.warning("Action buffer empty - holding position")

            # 4. 실행 프레임 소진 시 다음 추론 시작
            if (
                self.state.action_index >= self.config.execute_frames
                and not self.state.is_inferencing
            ):
                self._request_inference_async()

            self.state.step_count += 1

            # 제어 주파수 유지
            elapsed = time.time() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info(f"Leapfrog stopped after {self.state.step_count} steps")

    def stop(self):
        """제어 루프 정지."""
        self._running = False
        if self._inference_thread:
            self._inference_thread.join(timeout=5.0)
