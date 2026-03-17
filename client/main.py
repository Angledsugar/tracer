"""
DVA Main Entry Point
Isaac Sim + Cosmos를 연결하여 DVA 파이프라인 실행

중요: Isaac Sim의 SimulationApp은 torch 등 다른 GPU 라이브러리보다
먼저 초기화되어야 합니다. 따라서 torch/모델 import를 Isaac Sim 초기화
이후로 지연시킵니다.
"""

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="DVA: Direct Video-Action Model")
    parser.add_argument("--cosmos-server", type=str, default="localhost:50051",
                        help="Cosmos gRPC server address")
    parser.add_argument("--idm-checkpoint", type=str, default=None,
                        help="IDM model checkpoint path")
    parser.add_argument("--task", type=str, default="FrankaPickAndPlace",
                        help="Isaac Sim task name")
    parser.add_argument("--language", type=str, default="pick up the red block",
                        help="Language instruction for the task")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Maximum control steps")
    parser.add_argument("--control-hz", type=float, default=20.0,
                        help="Robot control frequency")
    parser.add_argument("--headless", action="store_true",
                        help="Run Isaac Sim in headless mode")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for IDM inference")
    args = parser.parse_args()

    # 1. Isaac Sim 환경 초기화 (반드시 torch import 전에 실행)
    logger.info("Initializing Isaac Sim environment...")
    from client.isaacsim_env import IsaacSimEnv

    env = IsaacSimEnv(
        task_name=args.task,
        headless=args.headless,
    )
    env.initialize()
    env.reset()

    # 2. torch 및 모델 import (Isaac Sim 초기화 이후)
    import torch
    from client.cosmos_client import CosmosClient
    from client.leapfrog_controller import LeapfrogController, LeapfrogConfig
    from models.inverse_dynamics import InverseDynamicsModel

    # 3. Cosmos 서버 연결
    logger.info(f"Connecting to Cosmos server at {args.cosmos_server}...")
    cosmos = CosmosClient(server_address=args.cosmos_server)
    health = cosmos.health_check()
    logger.info(f"Cosmos server: {health}")

    # 4. IDM 로드
    logger.info("Loading Inverse Dynamics Model...")
    idm = InverseDynamicsModel(action_dim=7, proprio_dim=7)
    if args.idm_checkpoint:
        idm.load_state_dict(torch.load(args.idm_checkpoint, map_location=args.device))
    idm.to(args.device)
    idm.eval()

    # 5. Leapfrog 컨트롤러 구성
    config = LeapfrogConfig(
        control_hz=args.control_hz,
        num_output_frames=12,
        execute_frames=8,
        overlap_frames=4,
    )

    controller = LeapfrogController(
        cosmos_client=cosmos,
        idm=idm,
        config=config,
        get_observation=env.get_observation,
        execute_action=env.execute_action,
        language_instruction=args.language,
        idm_device=args.device,
        show_predicted_frames=env.show_predicted_frames,
    )

    # 6. 실행
    logger.info("Starting DVA Leapfrog control loop...")
    try:
        controller.run(max_steps=args.max_steps)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        controller.stop()
        cosmos.close()
        env.close()

    logger.info("Done.")


if __name__ == "__main__":
    main()
