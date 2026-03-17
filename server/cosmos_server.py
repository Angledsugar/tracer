"""
Cosmos Predict 2.5 gRPC Server
GPU 서버에서 실행 - Action-Conditioned 비디오 생성 모델 서빙

로딩 우선순위:
1. Cosmos Predict 2.5 네이티브 (cosmos_predict2 + action_conditioned)
2. Placeholder 모델 (개발/테스트용)
"""

import os
import time
import io
import logging
from concurrent import futures
from typing import Optional

import grpc
import numpy as np
import torch
from PIL import Image

from proto import video_service_pb2
from proto import video_service_pb2_grpc

logger = logging.getLogger(__name__)


class CosmosModelWrapper:
    """Cosmos Predict 2.5 action-conditioned model wrapper."""

    def __init__(self, model_path: str, device: str = "cuda:0", cosmos25_repo: str = ""):
        self.device = device
        self.model = None
        self.model_path = model_path
        self.cosmos25_repo = cosmos25_repo
        self._load_model()

    def _load_model(self):
        """Load Cosmos action-conditioned model."""
        logger.info(f"Loading Cosmos model from {self.model_path}")

        if self.cosmos25_repo:
            try:
                self._load_cosmos25()
                return
            except Exception as e:
                logger.warning(f"Cosmos 2.5 loading failed: {e}")

        logger.warning("Using placeholder model.")
        self.model = PlaceholderModel(self.device)

    def _load_cosmos25(self):
        """Cosmos Predict 2.5 action-conditioned 모델 로드."""
        import sys
        # cosmos-predict2.5 저장소를 Python path에 추가
        if self.cosmos25_repo and self.cosmos25_repo not in sys.path:
            sys.path.insert(0, self.cosmos25_repo)

        from cosmos_predict2.action_conditioned_config import (
            ActionConditionedSetupArguments,
            DEFAULT_MODEL_KEY,
        )
        from cosmos_predict2.action_conditioned import inference as cosmos_inference

        # Setup arguments 구성
        setup_kwargs = {
            "output_dir": "outputs/cosmos25_server",
            "model": DEFAULT_MODEL_KEY.name,
        }
        if self.model_path and os.path.isdir(self.model_path):
            setup_kwargs["checkpoint_dir"] = self.model_path

        setup_args = ActionConditionedSetupArguments(**setup_kwargs)

        self.model = Cosmos25Model(
            setup_args=setup_args,
            cosmos_inference_fn=cosmos_inference,
            device=self.device,
            cosmos25_repo=self.cosmos25_repo,
        )
        logger.info("Cosmos Predict 2.5 action-conditioned model loaded")

    def predict(
        self,
        context_frames: list[np.ndarray],
        language: str,
        previous_actions: list[dict],
        num_output_frames: int = 12,
        guidance_scale: float = 7.0,
        num_denoise_steps: int = 35,
        seed: int = 42,
    ) -> list[np.ndarray]:
        return self.model.predict(
            context_frames=context_frames,
            language=language,
            previous_actions=previous_actions,
            num_output_frames=num_output_frames,
            guidance_scale=guidance_scale,
            num_denoise_steps=num_denoise_steps,
            seed=seed,
        )

    def get_gpu_memory_used(self) -> int:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(self.device) // (1024 * 1024)
        return 0


class Cosmos25Model:
    """Cosmos Predict 2.5 action-conditioned 모델.

    cosmos-predict2.5 저장소의 inference API를 직접 호출합니다.
    """

    def __init__(self, setup_args, cosmos_inference_fn, device: str, cosmos25_repo: str):
        self.setup_args = setup_args
        self.cosmos_inference_fn = cosmos_inference_fn
        self.device = device
        self.cosmos25_repo = cosmos25_repo
        self._pipeline = None

    def _ensure_pipeline(self):
        """Lazy loading: 첫 predict 호출 시 파이프라인 초기화."""
        if self._pipeline is not None:
            return

        import sys
        if self.cosmos25_repo and self.cosmos25_repo not in sys.path:
            sys.path.insert(0, self.cosmos25_repo)

        from cosmos_predict2._src.predict2.action.action_inference import Video2WorldCLI

        self._pipeline = Video2WorldCLI(self.setup_args)
        logger.info("Cosmos 2.5 pipeline initialized")

    def predict(
        self,
        context_frames: list[np.ndarray],
        language: str,
        previous_actions: list[dict],
        num_output_frames: int,
        guidance_scale: float,
        num_denoise_steps: int,
        seed: int,
    ) -> list[np.ndarray]:
        try:
            self._ensure_pipeline()
        except Exception as e:
            logger.warning(f"Pipeline init failed ({e}), using simple fallback")
            return self._fallback_predict(context_frames, num_output_frames, seed)

        # Action을 numpy 배열로 변환: (T, 7)
        actions_array = np.zeros((num_output_frames, 7), dtype=np.float32)
        if previous_actions:
            for i, a in enumerate(previous_actions[:num_output_frames]):
                actions_array[i] = [
                    a["dx"], a["dy"], a["dz"],
                    a["drx"], a["dry"], a["drz"],
                    a["gripper"],
                ]

        # Context frame 준비
        if not context_frames:
            return [np.zeros((256, 256, 3), dtype=np.uint8)] * num_output_frames

        initial_frame = context_frames[-1]

        try:
            # Cosmos 2.5 generate
            generated_video = self._pipeline.generate_vid2world(
                conditioned_frames=initial_frame,
                actions=actions_array,
                guidance=guidance_scale,
                num_steps=num_denoise_steps,
                seed=seed,
            )

            # 출력을 프레임 리스트로 변환
            return self._video_to_frames(generated_video, num_output_frames)

        except Exception as e:
            logger.error(f"Cosmos 2.5 inference failed: {e}")
            return self._fallback_predict(context_frames, num_output_frames, seed)

    def _video_to_frames(self, video, num_output_frames: int) -> list[np.ndarray]:
        """비디오 텐서/배열을 프레임 리스트로 변환."""
        if isinstance(video, torch.Tensor):
            video = video.cpu()
            if video.ndim == 4:  # (T, C, H, W)
                frames = []
                for i in range(min(video.shape[0], num_output_frames)):
                    frame = video[i].permute(1, 2, 0).numpy()
                    if frame.max() <= 1.0:
                        frame = (frame * 255).clip(0, 255)
                    frames.append(frame.astype(np.uint8))
                return frames
            elif video.ndim == 5:  # (B, T, C, H, W)
                return self._video_to_frames(video[0], num_output_frames)

        if isinstance(video, np.ndarray):
            if video.ndim == 4:  # (T, H, W, C) or (T, C, H, W)
                frames = []
                for i in range(min(video.shape[0], num_output_frames)):
                    frame = video[i]
                    if frame.shape[0] == 3:  # (C, H, W) → (H, W, C)
                        frame = np.transpose(frame, (1, 2, 0))
                    if frame.max() <= 1.0:
                        frame = (frame * 255).clip(0, 255)
                    frames.append(frame.astype(np.uint8))
                return frames

        if isinstance(video, list):
            return [np.array(f).astype(np.uint8) for f in video[:num_output_frames]]

        return self._fallback_predict([], num_output_frames, 42)

    def _fallback_predict(self, context_frames, num_output_frames, seed):
        """파이프라인 실패 시 간단한 fallback."""
        if not context_frames:
            return [np.zeros((256, 256, 3), dtype=np.uint8)] * num_output_frames
        rng = np.random.RandomState(seed)
        last = context_frames[-1].copy()
        frames = []
        for _ in range(num_output_frames):
            noise = rng.randint(-5, 5, last.shape, dtype=np.int16)
            frame = np.clip(last.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            frames.append(frame)
            last = frame
        return frames


class PlaceholderModel:
    """Cosmos 없이 개발/테스트용 placeholder 모델."""

    def __init__(self, device: str):
        self.device = device
        logger.info(f"PlaceholderModel initialized on {device}")

    def predict(self, context_frames, language, previous_actions,
                num_output_frames, guidance_scale, num_denoise_steps, seed):
        if not context_frames:
            return [np.zeros((256, 256, 3), dtype=np.uint8)] * num_output_frames

        rng = np.random.RandomState(seed)
        last_frame = context_frames[-1].copy()
        generated = []
        for i in range(num_output_frames):
            frame = last_frame.copy()
            noise = rng.randint(-5, 5, frame.shape, dtype=np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            generated.append(frame)
            last_frame = frame

        time.sleep(0.1)
        return generated


class CosmosVideoServicer(video_service_pb2_grpc.CosmosVideoServiceServicer):
    """gRPC service implementation."""

    def __init__(self, model: CosmosModelWrapper, save_dir: str = ""):
        self.model = model
        self._save_dir = save_dir
        self._request_count = 0
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            logger.info(f"Saving predicted frames to: {save_dir}")

    def _decode_frames(self, frame_bytes_list: list[bytes]) -> list[np.ndarray]:
        frames = []
        for fb in frame_bytes_list:
            img = Image.open(io.BytesIO(fb))
            frames.append(np.array(img))
        return frames

    def _encode_frame(self, frame: np.ndarray) -> bytes:
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    def _parse_actions(self, proto_actions) -> list[dict]:
        return [
            {
                "dx": a.dx, "dy": a.dy, "dz": a.dz,
                "drx": a.drx, "dry": a.dry, "drz": a.drz,
                "gripper": a.gripper,
            }
            for a in proto_actions
        ]

    def _save_frames(
        self,
        context_frames: list[np.ndarray],
        generated_frames: list[np.ndarray],
    ):
        req_dir = os.path.join(self._save_dir, f"request_{self._request_count:06d}")
        os.makedirs(req_dir, exist_ok=True)

        if context_frames:
            img = Image.fromarray(context_frames[-1])
            img.save(os.path.join(req_dir, "context.jpg"), quality=95)

        for i, frame in enumerate(generated_frames):
            img = Image.fromarray(frame)
            img.save(os.path.join(req_dir, f"predicted_{i:03d}.jpg"), quality=95)

        logger.info(f"Saved {len(generated_frames)} frames to {req_dir}")
        self._request_count += 1

    def PredictVideo(self, request, context):
        start_time = time.time()

        context_frames = self._decode_frames(request.context_frames)
        previous_actions = self._parse_actions(request.previous_actions)

        num_output = request.num_output_frames or 12
        guidance = request.guidance_scale or 7.0
        steps = request.num_denoise_steps or 35
        seed = request.seed or 42

        logger.info(
            f"PredictVideo: {len(context_frames)} context frames, "
            f"{num_output} output frames, guidance={guidance}, steps={steps}"
        )

        generated_frames = self.model.predict(
            context_frames=context_frames,
            language=request.language_instruction,
            previous_actions=previous_actions,
            num_output_frames=num_output,
            guidance_scale=guidance,
            num_denoise_steps=steps,
            seed=seed,
        )

        if self._save_dir:
            self._save_frames(context_frames, generated_frames)

        encoded_frames = [self._encode_frame(f) for f in generated_frames]
        elapsed_ms = (time.time() - start_time) * 1000

        logger.info(f"PredictVideo completed in {elapsed_ms:.1f}ms")

        return video_service_pb2.PredictResponse(
            generated_frames=encoded_frames,
            inference_time_ms=elapsed_ms,
        )

    def PredictVideoStream(self, request, context):
        context_frames = self._decode_frames(request.context_frames)
        previous_actions = self._parse_actions(request.previous_actions)

        num_output = request.num_output_frames or 12
        guidance = request.guidance_scale or 7.0
        steps = request.num_denoise_steps or 35
        seed = request.seed or 42

        generated_frames = self.model.predict(
            context_frames=context_frames,
            language=request.language_instruction,
            previous_actions=previous_actions,
            num_output_frames=num_output,
            guidance_scale=guidance,
            num_denoise_steps=steps,
            seed=seed,
        )

        for i, frame in enumerate(generated_frames):
            yield video_service_pb2.FrameResponse(
                frame=self._encode_frame(frame),
                frame_index=i,
            )

    def HealthCheck(self, request, context):
        return video_service_pb2.HealthResponse(
            ready=True,
            model_name="cosmos-predict2.5-action-conditioned",
            gpu_memory_used_mb=self.model.get_gpu_memory_used(),
        )


def serve(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 50051,
    max_workers: int = 4,
    device: str = "cuda:0",
    save_dir: str = "",
    cosmos25_repo: str = "",
):
    """Start the gRPC server."""
    logging.basicConfig(level=logging.INFO)

    model = CosmosModelWrapper(
        model_path=model_path, device=device, cosmos25_repo=cosmos25_repo,
    )
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ("grpc.max_send_message_length", 256 * 1024 * 1024),
            ("grpc.max_receive_message_length", 256 * 1024 * 1024),
        ],
    )
    video_service_pb2_grpc.add_CosmosVideoServiceServicer_to_server(
        CosmosVideoServicer(model, save_dir=save_dir), server
    )
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    logger.info(f"Cosmos server started on {host}:{port}")
    server.wait_for_termination()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Cosmos Predict 2.5 Video Generation Server")
    parser.add_argument("--model-path", type=str, default="",
                        help="Checkpoint directory or HuggingFace model ID")
    parser.add_argument("--cosmos25-repo", type=str, default="",
                        help="Path to cosmos-predict2.5 repository (e.g. ~/ccy/cosmos-predict2.5)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save-dir", type=str, default="",
                        help="Directory to save predicted frames (empty = disabled)")
    args = parser.parse_args()

    serve(
        model_path=args.model_path, host=args.host, port=args.port,
        device=args.device, save_dir=args.save_dir, cosmos25_repo=args.cosmos25_repo,
    )


if __name__ == "__main__":
    main()
