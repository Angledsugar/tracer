"""
Cosmos Predict 2.5 gRPC Server
RTX 4090 서버에서 실행 - 비디오 생성 모델 서빙
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

# Proto generated files
import sys
sys.path.append("..")
from proto import video_service_pb2
from proto import video_service_pb2_grpc

logger = logging.getLogger(__name__)


class CosmosModelWrapper:
    """Cosmos Predict 2.5 action-conditioned model wrapper."""

    def __init__(self, model_path: str, device: str = "cuda:0"):
        self.device = device
        self.model = None
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """Load Cosmos action-conditioned model."""
        logger.info(f"Loading Cosmos model from {self.model_path}")

        # TODO: Replace with actual Cosmos model loading
        # from cosmos_predict2.inference import Video2WorldInference
        # self.model = Video2WorldInference(
        #     config_path=self.model_path,
        #     device=self.device,
        # )

        # Placeholder for development
        logger.warning("Using PLACEHOLDER model - replace with actual Cosmos loading")
        self.model = PlaceholderModel(self.device)

        logger.info("Model loaded successfully")

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


class PlaceholderModel:
    """Placeholder model for development/testing without Cosmos."""

    def __init__(self, device: str):
        self.device = device
        logger.info(f"PlaceholderModel initialized on {device}")

    def predict(self, context_frames, language, previous_actions,
                num_output_frames, guidance_scale, num_denoise_steps, seed):
        """Generate dummy future frames by slightly modifying the last context frame."""
        if not context_frames:
            # Return black frames
            return [np.zeros((256, 256, 3), dtype=np.uint8)] * num_output_frames

        last_frame = context_frames[-1].copy()
        generated = []
        for i in range(num_output_frames):
            # Simulate small changes
            frame = last_frame.copy()
            noise = np.random.randint(-5, 5, frame.shape, dtype=np.int16)
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
            model_name="cosmos-predict2.5-2b-robot-action-cond",
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
            ("grpc.max_send_message_length", 256 * 1024 * 1024),  # 256MB
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cosmos Video Generation Server")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    serve(model_path=args.model_path, host=args.host, port=args.port, device=args.device)
