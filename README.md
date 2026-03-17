# Tracer

DVA (Direct Video-Action Model) 파이프라인 — 비디오 생성 모델과 역동역학 모델을 결합하여 로봇을 제어하는 시스템입니다.

## 개요

```
┌──────────────────────────────────────────────────────────────┐
│  Client (로컬)                                               │
│  ┌──────────┐   ┌─────────────┐   ┌────────────────────┐    │
│  │ Isaac Sim │──▶│  Leapfrog   │──▶│  IDM (Inverse      │    │
│  │ 환경      │◀──│  Controller │◀──│  Dynamics Model)   │    │
│  └──────────┘   └──────┬──────┘   └────────────────────┘    │
│                        │ gRPC                                │
├────────────────────────┼─────────────────────────────────────┤
│  Server (GPU)          │                                     │
│  ┌─────────────────────┴───────────────────────────────┐    │
│  │  Cosmos Predict 2 (Action-Conditioned Video Gen)    │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

### 파이프라인 흐름

1. **Isaac Sim** — 로봇 시뮬레이션 환경에서 현재 프레임(RGB 이미지) 캡처
2. **Cosmos** — 현재 프레임 + 언어 명령을 받아 미래 비디오 프레임 생성 (gRPC 통신)
3. **IDM** — 생성된 프레임 쌍을 7차원 로봇 액션(dx, dy, dz, drx, dry, drz, gripper)으로 변환
4. **Leapfrog Controller** — 추론과 실행을 오버랩하여 끊김 없는 연속 제어 수행

## 프로젝트 구조

```
tracer/
├── client/
│   ├── main.py                  # DVA 파이프라인 진입점
│   ├── cosmos_client.py         # Cosmos gRPC 클라이언트
│   ├── isaacsim_env.py          # Isaac Sim 환경 래퍼
│   └── leapfrog_controller.py   # Leapfrog 추론/실행 컨트롤러
├── server/
│   └── cosmos_server.py         # Cosmos 모델 gRPC 서버
├── models/
│   └── inverse_dynamics.py      # Inverse Dynamics Model (ResNet 기반)
├── proto/
│   ├── video_service.proto      # gRPC 서비스 정의
│   ├── video_service_pb2.py     # 생성된 protobuf 코드
│   └── video_service_pb2_grpc.py
├── scripts/
│   └── compile_proto.sh         # Proto 컴파일 스크립트
└── pyproject.toml               # 프로젝트 설정 및 의존성
```

## 설치

### 요구사항

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/) (패키지 매니저)
- CUDA 지원 GPU (IDM 추론 및 Cosmos 서버)

### 기본 설치

```bash
git clone https://github.com/Angledsugar/tracer.git
cd tracer
uv sync
```

### Isaac Sim (선택)

Isaac Sim은 NVIDIA Omniverse를 통해 별도로 설치합니다. 설치되지 않은 환경에서는 자동으로 placeholder 모드로 동작합니다.

- [Isaac Sim 설치 가이드](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_python.html)

### Cosmos 모델 (선택)

Cosmos Predict 2 모델은 두 가지 방법으로 사용할 수 있습니다:

**방법 1: 네이티브 라이브러리**
```bash
uv pip install "cosmos-predict2[cu126]" --extra-index-url https://nvidia-cosmos.github.io/cosmos-dependencies/cu126_torch260/simple
```

**방법 2: HuggingFace Diffusers**
```bash
uv pip install diffusers transformers accelerate
```

모델이 설치되지 않으면 서버는 placeholder 모드로 동작합니다.

## 실행

### 1. Cosmos 서버 시작

```bash
# 실제 모델 사용
uv run python -m server.cosmos_server --model-path <모델 경로 또는 HuggingFace ID>

# 예시 (HuggingFace)
uv run python -m server.cosmos_server --model-path nvidia/Cosmos-Predict2-2B-Video2World --device cuda:0
```

### 2. 클라이언트 실행

```bash
uv run python -m client.main \
    --cosmos-server localhost:50051 \
    --task FrankaPickAndPlace \
    --language "pick up the red block" \
    --device cuda:0
```

### 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--cosmos-server` | `localhost:50051` | Cosmos gRPC 서버 주소 |
| `--idm-checkpoint` | `None` | IDM 모델 체크포인트 경로 |
| `--task` | `FrankaPickAndPlace` | Isaac Sim 작업 이름 |
| `--language` | `pick up the red block` | 언어 명령 |
| `--max-steps` | `1000` | 최대 제어 스텝 수 |
| `--control-hz` | `20.0` | 로봇 제어 주파수 |
| `--headless` | `False` | Isaac Sim headless 모드 |
| `--device` | `cuda:0` | IDM 추론 디바이스 |

## Proto 재컴파일

gRPC proto 정의를 수정한 경우:

```bash
bash scripts/compile_proto.sh
```

## 기술 스택

- **PyTorch** / **torchvision** — IDM 모델 및 추론
- **gRPC** / **Protobuf** — 클라이언트-서버 통신
- **Isaac Sim** — 로봇 시뮬레이션
- **Cosmos Predict 2** — Action-Conditioned 비디오 생성
