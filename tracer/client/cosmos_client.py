"""
Cosmos gRPC Client
로컬 머신에서 원격 Cosmos 서버와 통신
"""

import io
import time
import logging
from typing import Optional

import grpc
import numpy as np
from PIL import Image

import sys
sys.path.append("..")
from proto import video_service_pb2
from proto import video_service_pb2_grpc

logger = logging.getLogger(__name__)


class CosmosClient:
    """gRPC client for remote Cosmos video generation server."""

    def __init__(self, server_address: str = "localhost:50051", timeout: float = 30.0):
        self.server_address = server_address
        self.timeout = timeout
        self.channel = None
        self.stub = None
        self._connect()

    def _connect(self):
        """Establish gRPC connection."""
        self.channel = grpc.insecure_channel(
            self.server_address,
            options=[
                ("grpc.max_send_message_length", 256 * 1024 * 1024),
                ("grpc.max_receive_message_length", 256 * 1024 * 1024),
            ],
        )
        self.stub = video_service_pb2_grpc.CosmosVideoServiceStub(self.channel)
        logger.info(f"Connected to Cosmos server at {self.server_address}")

    def _encode_frame(self, frame: np.ndarray) -> bytes:
        """Encode numpy array to JPEG bytes."""
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    def _decode_frame(self, frame_bytes: bytes) -> np.ndarray:
        """Decode JPEG bytes to numpy array."""
        img = Image.open(io.BytesIO(frame_bytes))
        return np.array(img)

    def predict(
        self,
        context_frames: list[np.ndarray],
        language_instruction: str = "",
        previous_actions: Optional[list[dict]] = None,
        num_output_frames: int = 12,
        guidance_scale: float = 7.0,
        num_denoise_steps: int = 35,
        seed: int = 42,
    ) -> tuple[list[np.ndarray], float]:
        """
        Request video prediction from remote Cosmos server.

        Returns:
            (generated_frames, inference_time_ms)
        """
        # Encode context frames
        encoded_frames = [self._encode_frame(f) for f in context_frames]

        # Build action messages
        proto_actions = []
        if previous_actions:
            for a in previous_actions:
                proto_actions.append(video_service_pb2.Action(
                    dx=a.get("dx", 0), dy=a.get("dy", 0), dz=a.get("dz", 0),
                    drx=a.get("drx", 0), dry=a.get("dry", 0), drz=a.get("drz", 0),
                    gripper=a.get("gripper", 0),
                ))

        request = video_service_pb2.PredictRequest(
            context_frames=encoded_frames,
            language_instruction=language_instruction,
            previous_actions=proto_actions,
            num_output_frames=num_output_frames,
            guidance_scale=guidance_scale,
            num_denoise_steps=num_denoise_steps,
            seed=seed,
        )

        response = self.stub.PredictVideo(request, timeout=self.timeout)

        generated_frames = [self._decode_frame(f) for f in response.generated_frames]
        return generated_frames, response.inference_time_ms

    def predict_stream(
        self,
        context_frames: list[np.ndarray],
        language_instruction: str = "",
        previous_actions: Optional[list[dict]] = None,
        num_output_frames: int = 12,
        guidance_scale: float = 7.0,
        num_denoise_steps: int = 35,
        seed: int = 42,
    ):
        """Streaming prediction - yields frames as they arrive."""
        encoded_frames = [self._encode_frame(f) for f in context_frames]

        proto_actions = []
        if previous_actions:
            for a in previous_actions:
                proto_actions.append(video_service_pb2.Action(
                    dx=a.get("dx", 0), dy=a.get("dy", 0), dz=a.get("dz", 0),
                    drx=a.get("drx", 0), dry=a.get("dry", 0), drz=a.get("drz", 0),
                    gripper=a.get("gripper", 0),
                ))

        request = video_service_pb2.PredictRequest(
            context_frames=encoded_frames,
            language_instruction=language_instruction,
            previous_actions=proto_actions,
            num_output_frames=num_output_frames,
            guidance_scale=guidance_scale,
            num_denoise_steps=num_denoise_steps,
            seed=seed,
        )

        for response in self.stub.PredictVideoStream(request, timeout=self.timeout):
            frame = self._decode_frame(response.frame)
            yield frame, response.frame_index

    def health_check(self) -> dict:
        """Check server health."""
        response = self.stub.HealthCheck(
            video_service_pb2.Empty(), timeout=5.0
        )
        return {
            "ready": response.ready,
            "model_name": response.model_name,
            "gpu_memory_used_mb": response.gpu_memory_used_mb,
        }

    def close(self):
        if self.channel:
            self.channel.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
