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
        if self.cosmos25_repo and self.cosmos25_repo not in sys.path:
            sys.path.insert(0, self.cosmos25_repo)

        from cosmos_predict2.action_conditioned_config import (
            ActionConditionedSetupArguments,
            DEFAULT_MODEL_KEY,
        )
        from cosmos_predict2.config import MODEL_CHECKPOINTS
        from cosmos_predict2._src.predict2.inference.video2world import (
            Video2WorldInference,
        )

        # Setup arguments 구성
        setup_kwargs = {
            "output_dir": "outputs/cosmos25_server",
            "model": DEFAULT_MODEL_KEY.name,
        }
        if self.model_path and os.path.isdir(self.model_path):
            setup_kwargs["checkpoint_dir"] = self.model_path

        setup_args = ActionConditionedSetupArguments(**setup_kwargs)

        # 체크포인트 및 실험 설정 해석
        checkpoint = MODEL_CHECKPOINTS[setup_args.model_key]
        experiment = setup_args.experiment or checkpoint.experiment
        checkpoint_path = setup_args.checkpoint_path or checkpoint.s3.uri

        # action-conditioned 전용 config (video2world 기본 config가 아님!)
        config_file = "cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py"
        original_cwd = os.getcwd()
        if self.cosmos25_repo:
            os.chdir(self.cosmos25_repo)

        logger.info(
            f"Initializing Video2WorldInference: "
            f"experiment={experiment}, config_file={config_file}"
        )

        # VLMBaseModel (Reason1-7B, ~14GB)을 로딩 단계부터 CPU에 유지
        # 이 모델이 GPU에 올라가면 VAE+DiT와 합쳐서 ~23GB → OOM
        from cosmos_predict2._src.reason1.models.vlm_base import VLMBaseModel

        _original_vlm_init = VLMBaseModel.__init__

        def _patched_vlm_init(self_vlm, *args, **kwargs):
            _original_vlm_init(self_vlm, *args, **kwargs)
            # 즉시 CPU로 이동 (GPU에 올라간 직후 바로 내림)
            self_vlm.to('cpu')
            torch.cuda.empty_cache()
            logger.info("VLMBaseModel (Reason1-7B) kept on CPU during loading")

        VLMBaseModel.__init__ = _patched_vlm_init

        try:
            pipeline = Video2WorldInference(
                experiment_name=experiment,
                ckpt_path=checkpoint_path,
                s3_credential_path="",
                context_parallel_size=1,
                config_file=config_file,
            )
        finally:
            os.chdir(original_cwd)
            VLMBaseModel.__init__ = _original_vlm_init

        self.model = Cosmos25Model(pipeline=pipeline, device=self.device)
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

    텍스트 임베딩을 한번 계산/캐싱 후 텍스트 인코더(~14GB)를 해제하여
    24GB GPU에서도 추론 가능하게 합니다.
    """

    def __init__(self, pipeline, device: str):
        self._pipeline = pipeline
        self.device = device
        self._cached_text_emb = None
        self._cached_neg_text_emb = None
        self._text_encoder_freed = False

    def cache_prompt_and_free_encoder(
        self, prompt: str, negative_prompt: str = "The video captures a series of frames showing ugly scenes."
    ):
        """프롬프트 임베딩을 캐싱하고 텍스트 인코더를 GPU에서 완전 해제."""
        if self._text_encoder_freed:
            return

        text_encoder = self._pipeline.model.text_encoder
        if text_encoder is None:
            logger.warning("No text encoder found, skipping cache")
            return

        logger.info(f"Computing text embedding for: '{prompt}'")

        # 1. 텍스트 임베딩 계산 (GPU에서)
        self._cached_text_emb = text_encoder.compute_text_embeddings_online(
            data_batch={"ai_caption": [prompt], "images": None},
            input_caption_key="ai_caption",
        )
        self._cached_neg_text_emb = text_encoder.compute_text_embeddings_online(
            data_batch={"ai_caption": [negative_prompt], "images": None},
            input_caption_key="ai_caption",
        )

        # GPU에 유지 (추론 시 사용)
        self._cached_text_emb = self._cached_text_emb.cuda().to(torch.bfloat16)
        self._cached_neg_text_emb = self._cached_neg_text_emb.cuda().to(torch.bfloat16)

        logger.info(f"Cached text embedding shape: {self._cached_text_emb.shape}")

        # 2. 텍스트 인코더 완전 해제
        param_gb = sum(
            p.numel() * p.element_size() for p in text_encoder.parameters()
        ) / (1024**3)

        self._pipeline.model.text_encoder = None
        del text_encoder
        torch.cuda.empty_cache()

        gpu_free = torch.cuda.mem_get_info(self.device)[0] / (1024**3)
        logger.info(f"Freed text encoder (~{param_gb:.1f} GB). GPU available: {gpu_free:.1f} GiB")

        # 3. get_text_embedding을 monkey-patch하여 캐싱된 임베딩 반환
        #    generate_vid2world()에서 text_encoder가 None이면
        #    get_text_embedding(prompt)을 호출함 → 캐싱된 값 반환
        cached_emb = self._cached_text_emb
        cached_neg_emb = self._cached_neg_text_emb

        import cosmos_predict2._src.predict2.inference.video2world as v2w_module
        original_get_text_embedding = v2w_module.get_text_embedding

        def _cached_get_text_embedding(text):
            """캐싱된 텍스트 임베딩을 반환."""
            logger.debug(f"Using cached embedding for: '{text[:50]}...'")
            if "ugly" in text.lower() or "static" in text.lower():
                return cached_neg_emb
            return cached_emb

        v2w_module.get_text_embedding = _cached_get_text_embedding
        logger.info("Patched get_text_embedding with cached embeddings")

        self._text_encoder_freed = True

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
        import torchvision

        if not context_frames:
            return [np.zeros((256, 256, 3), dtype=np.uint8)] * num_output_frames

        # Context frame → tensor 준비 (이미지만 입력, action 없음)
        img_array = context_frames[-1]  # (H, W, 3) uint8
        img_tensor = torchvision.transforms.functional.to_tensor(img_array).unsqueeze(0)  # (1, 3, H, W)
        num_video_frames = num_output_frames + 1  # 조건 프레임 1 + 생성 프레임 N

        # (B, C, T, H, W) 형식의 비디오 입력 생성
        vid_input = torch.cat(
            [img_tensor, torch.zeros_like(img_tensor).repeat(num_video_frames - 1, 1, 1, 1)],
            dim=0,
        )
        vid_input = (vid_input * 255.0).to(torch.uint8)
        vid_input = vid_input.unsqueeze(0).permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)

        # 빈 action (이미지 + 언어만으로 예측)
        zero_actions = torch.zeros(num_output_frames, 7)

        try:
            video = self._pipeline.generate_vid2world(
                prompt=language or "",
                input_path=vid_input,
                action=zero_actions,
                guidance=guidance_scale,
                num_video_frames=num_video_frames,
                num_latent_conditional_frames=1,
                resolution="none",
                seed=seed,
                negative_prompt="The video captures a series of frames showing ugly scenes.",
                num_steps=num_denoise_steps,
            )

            return self._video_to_frames(video, num_output_frames)

        except Exception as e:
            logger.error(f"Cosmos 2.5 inference failed: {e}")
            return self._fallback_predict(context_frames, num_output_frames, seed)

    def _video_to_frames(self, video, num_output_frames: int) -> list[np.ndarray]:
        """비디오 텐서를 프레임 리스트로 변환. 출력 범위: [-1, 1] → [0, 255]."""
        if isinstance(video, torch.Tensor):
            video = video.cpu().float()

            # (B, C, T, H, W) → (T, C, H, W)
            if video.ndim == 5:
                video = video[0]  # batch 제거
            if video.ndim == 4 and video.shape[0] == 3:
                # (C, T, H, W) → (T, C, H, W)
                video = video.permute(1, 0, 2, 3)

            # [-1, 1] → [0, 1]
            video_normalized = (video - video.min()) / (video.max() - video.min() + 1e-8)

            frames = []
            # 첫 프레임은 조건 프레임이므로 건너뜀
            start = 1 if video_normalized.shape[0] > num_output_frames else 0
            for i in range(start, min(video_normalized.shape[0], start + num_output_frames)):
                frame = video_normalized[i].permute(1, 2, 0).numpy()  # (H, W, C)
                frame = (frame * 255).clip(0, 255).astype(np.uint8)
                frames.append(frame)
            return frames

        if isinstance(video, np.ndarray):
            video_norm = (video - video.min()) / (video.max() - video.min() + 1e-8)
            video_uint8 = (video_norm * 255).clip(0, 255).astype(np.uint8)
            if video_uint8.ndim == 4:
                return [video_uint8[i] for i in range(min(video_uint8.shape[0], num_output_frames))]

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
    prompt: str = "",
):
    """Start the gRPC server."""
    logging.basicConfig(level=logging.INFO)

    model = CosmosModelWrapper(
        model_path=model_path, device=device, cosmos25_repo=cosmos25_repo,
    )

    # 프롬프트 캐싱 + 텍스트 인코더 해제 (~14GB GPU 메모리 확보)
    if prompt and hasattr(model.model, 'cache_prompt_and_free_encoder'):
        model.model.cache_prompt_and_free_encoder(prompt)
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
    parser.add_argument("--prompt", type=str, default="",
                        help="Fixed prompt to cache (frees text encoder ~14GB after caching)")
    args = parser.parse_args()

    serve(
        model_path=args.model_path, host=args.host, port=args.port,
        device=args.device, save_dir=args.save_dir, cosmos25_repo=args.cosmos25_repo,
        prompt=args.prompt,
    )


if __name__ == "__main__":
    main()
