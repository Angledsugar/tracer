"""
Cosmos Predict 2 gRPC Server
RTX 4090 서버에서 실행 - Action-Conditioned 비디오 생성 모델 서빙

두 가지 모드를 지원합니다:
1. 실제 Cosmos 모델 (cosmos_predict2 패키지 필요)
2. Placeholder 모델 (개발/테스트용, 의존성 없음)
"""

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
    """Cosmos Predict 2 action-conditioned model wrapper."""

    def __init__(self, model_path: str, device: str = "cuda:0"):
        self.device = device
        self.model = None
        self.pipeline = None
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """Load Cosmos action-conditioned model."""
        logger.info(f"Loading Cosmos model from {self.model_path}")

        try:
            self._load_cosmos_native()
        except ImportError:
            try:
                self._load_cosmos_diffusers()
            except ImportError:
                logger.warning(
                    "Cosmos not available - using placeholder model. "
                    "Install cosmos-predict2 or diffusers to use the real model."
                )
                self.model = PlaceholderModel(self.device)

    def _load_cosmos_native(self):
        """cosmos_predict2 네이티브 라이브러리로 모델 로드."""
        from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
        from cosmos_predict2.configs.base.config_video2world import (
            get_cosmos_predict2_video2world_pipeline,
        )

        config = get_cosmos_predict2_video2world_pipeline(model_size="2B")
        self.pipeline = Video2WorldPipeline.from_config(
            config=config,
            dit_path=self.model_path,
        )
        self.model = NativeCosmosModel(self.pipeline, self.device)
        logger.info("Cosmos model loaded (native cosmos_predict2)")

    def _load_cosmos_diffusers(self):
        """HuggingFace Diffusers로 모델 로드."""
        from diffusers import Cosmos2VideoToWorldPipeline

        self.pipeline = Cosmos2VideoToWorldPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
        )
        self.pipeline.to(self.device)
        self.model = DiffusersCosmosModel(self.pipeline, self.device)
        logger.info("Cosmos model loaded (HuggingFace Diffusers)")

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
        """
        Generate future video frames.

        Args:
            context_frames: List of RGB frames [H, W, 3] uint8
            language: Task instruction
            previous_actions: List of {dx,dy,dz,drx,dry,drz,gripper}
            num_output_frames: Number of frames to generate
            guidance_scale: Classifier-free guidance scale
            num_denoise_steps: Number of denoising steps
            seed: Random seed

        Returns:
            List of generated RGB frames [H, W, 3] uint8
        """
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
        """Get GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(self.device) // (1024 * 1024)
        return 0


class NativeCosmosModel:
    """cosmos_predict2 네이티브 라이브러리를 사용하는 모델."""

    def __init__(self, pipeline, device: str):
        self.pipeline = pipeline
        self.device = device

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
        # Action을 numpy 배열로 변환: (T, 7)
        actions_array = None
        if previous_actions:
            actions_array = np.array([
                [a["dx"], a["dy"], a["dz"], a["drx"], a["dry"], a["drz"], a["gripper"]]
                for a in previous_actions
            ])

        # 마지막 context frame을 PIL Image로 변환
        last_frame = Image.fromarray(context_frames[-1])

        # 비디오 생성
        video = self.pipeline(
            input_path=last_frame,
            prompt=language,
            actions=actions_array,
            num_frames=num_output_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_denoise_steps,
            seed=seed,
        )

        # 출력을 numpy 프레임 리스트로 변환
        if isinstance(video, torch.Tensor):
            # (T, C, H, W) → list of (H, W, 3) uint8
            frames = []
            for i in range(video.shape[0]):
                frame = video[i].permute(1, 2, 0).cpu().numpy()
                frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
                frames.append(frame)
            return frames

        # 이미 numpy 배열인 경우
        if isinstance(video, np.ndarray):
            return [video[i] for i in range(video.shape[0])]

        return video


class DiffusersCosmosModel:
    """HuggingFace Diffusers를 사용하는 모델."""

    def __init__(self, pipeline, device: str):
        self.pipeline = pipeline
        self.device = device

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
        # Context frame을 PIL Image로 변환
        input_image = Image.fromarray(context_frames[-1])

        generator = torch.Generator(device=self.device).manual_seed(seed)

        output = self.pipeline(
            image=input_image,
            prompt=language,
            negative_prompt="static with no motion, blurry, low quality",
            num_inference_steps=num_denoise_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        # frames[0]은 PIL Image 리스트
        generated_frames = []
        for frame in output.frames[0][:num_output_frames]:
            if isinstance(frame, Image.Image):
                generated_frames.append(np.array(frame))
            else:
                generated_frames.append(frame)

        return generated_frames


class PlaceholderModel:
    """Cosmos 없이 개발/테스트용 placeholder 모델."""

    def __init__(self, device: str):
        self.device = device
        logger.info(f"PlaceholderModel initialized on {device}")

    def predict(self, context_frames, language, previous_actions,
                num_output_frames, guidance_scale, num_denoise_steps, seed):
        """Generate dummy future frames by slightly modifying the last context frame."""
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

        # Simulate inference time
        time.sleep(0.1)
        return generated


class CosmosVideoServicer(video_service_pb2_grpc.CosmosVideoServiceServicer):
    """gRPC service implementation."""

    def __init__(self, model: CosmosModelWrapper):
        self.model = model

    def _decode_frames(self, frame_bytes_list: list[bytes]) -> list[np.ndarray]:
        """Decode JPEG bytes to numpy arrays."""
        frames = []
        for fb in frame_bytes_list:
            img = Image.open(io.BytesIO(fb))
            frames.append(np.array(img))
        return frames

    def _encode_frame(self, frame: np.ndarray) -> bytes:
        """Encode numpy array to JPEG bytes."""
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    def _parse_actions(self, proto_actions) -> list[dict]:
        """Convert proto Action messages to dicts."""
        return [
            {
                "dx": a.dx, "dy": a.dy, "dz": a.dz,
                "drx": a.drx, "dry": a.dry, "drz": a.drz,
                "gripper": a.gripper,
            }
            for a in proto_actions
        ]

    def PredictVideo(self, request, context):
        """Handle single prediction request."""
        start_time = time.time()

        # Decode inputs
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

        # Run inference
        generated_frames = self.model.predict(
            context_frames=context_frames,
            language=request.language_instruction,
            previous_actions=previous_actions,
            num_output_frames=num_output,
            guidance_scale=guidance,
            num_denoise_steps=steps,
            seed=seed,
        )

        # Encode output
        encoded_frames = [self._encode_frame(f) for f in generated_frames]
        elapsed_ms = (time.time() - start_time) * 1000

        logger.info(f"PredictVideo completed in {elapsed_ms:.1f}ms")

        return video_service_pb2.PredictResponse(
            generated_frames=encoded_frames,
            inference_time_ms=elapsed_ms,
        )

    def PredictVideoStream(self, request, context):
        """Handle streaming prediction - send frames as they're generated."""
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
            model_name="cosmos-predict2-2b-action-conditioned",
            gpu_memory_used_mb=self.model.get_gpu_memory_used(),
        )


def serve(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 50051,
    max_workers: int = 4,
    device: str = "cuda:0",
):
    """Start the gRPC server."""
    logging.basicConfig(level=logging.INFO)

    model = CosmosModelWrapper(model_path=model_path, device=device)
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ("grpc.max_send_message_length", 256 * 1024 * 1024),
            ("grpc.max_receive_message_length", 256 * 1024 * 1024),
        ],
    )
    video_service_pb2_grpc.add_CosmosVideoServiceServicer_to_server(
        CosmosVideoServicer(model), server
    )
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    logger.info(f"Cosmos server started on {host}:{port}")
    server.wait_for_termination()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Cosmos Video Generation Server")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to Cosmos model checkpoint or HuggingFace model ID")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    serve(model_path=args.model_path, host=args.host, port=args.port, device=args.device)


if __name__ == "__main__":
    main()
