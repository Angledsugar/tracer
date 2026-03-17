"""
Inverse Dynamics Model (IDM)
비디오 프레임 쌍 → 로봇 행동 변환
"""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


class InverseDynamicsModel(nn.Module):
    """
    Given (current_frame, next_frame, proprioception),
    predict the action that transitions between them.

    Output: 7-dim action (dx, dy, dz, drx, dry, drz, gripper)
    """

    def __init__(
        self,
        action_dim: int = 7,
        proprio_dim: int = 7,
        hidden_dim: int = 256,
        backbone: str = "resnet18",
    ):
        super().__init__()

        # Visual encoder (shared weights for both frames)
        if backbone == "resnet18":
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.visual_encoder = nn.Sequential(*list(resnet.children())[:-1])
            visual_feat_dim = 512
        elif backbone == "resnet34":
            resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            self.visual_encoder = nn.Sequential(*list(resnet.children())[:-1])
            visual_feat_dim = 512
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Feature fusion and action prediction
        # Two frames (current + next) + proprioception
        fusion_dim = visual_feat_dim * 2 + proprio_dim

        self.action_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def encode_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """Encode a single frame. Input: [B, 3, H, W], Output: [B, feat_dim]"""
        feat = self.visual_encoder(frame)
        return feat.flatten(1)

    def forward(
        self,
        current_frame: torch.Tensor,
        next_frame: torch.Tensor,
        proprioception: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            current_frame: [B, 3, H, W] normalized to [0, 1]
            next_frame: [B, 3, H, W] normalized to [0, 1]
            proprioception: [B, proprio_dim] current joint state

        Returns:
            action: [B, 7] (dx, dy, dz, drx, dry, drz, gripper)
        """
        feat_current = self.encode_frame(current_frame)
        feat_next = self.encode_frame(next_frame)

        fused = torch.cat([feat_current, feat_next, proprioception], dim=-1)
        action = self.action_head(fused)
        return action

    @torch.no_grad()
    def predict(
        self,
        current_frame: np.ndarray,
        next_frame: np.ndarray,
        proprioception: np.ndarray,
        device: str = "cuda:0",
    ) -> np.ndarray:
        """
        Numpy interface for inference.

        Args:
            current_frame: [H, W, 3] uint8
            next_frame: [H, W, 3] uint8
            proprioception: [7] float

        Returns:
            action: [7] float
        """
        self.eval()

        def to_tensor(img):
            t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
            return t.unsqueeze(0).to(device)

        current_t = to_tensor(current_frame)
        next_t = to_tensor(next_frame)
        proprio_t = torch.from_numpy(proprioception).float().unsqueeze(0).to(device)

        action = self.forward(current_t, next_t, proprio_t)
        return action.cpu().numpy()[0]

    @torch.no_grad()
    def predict_chunk(
        self,
        current_frame: np.ndarray,
        future_frames: list[np.ndarray],
        proprioception: np.ndarray,
        device: str = "cuda:0",
    ) -> list[np.ndarray]:
        """
        Predict actions for a sequence of future frames.

        Args:
            current_frame: [H, W, 3] uint8 - starting frame
            future_frames: List of [H, W, 3] uint8 - generated future frames
            proprioception: [7] float - current proprioception

        Returns:
            List of [7] float actions
        """
        actions = []
        prev_frame = current_frame

        for next_frame in future_frames:
            action = self.predict(prev_frame, next_frame, proprioception, device)
            actions.append(action)
            prev_frame = next_frame

        return actions
