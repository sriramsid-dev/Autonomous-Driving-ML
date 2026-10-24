"""
Module C: Lane Detection
Algorithm: Ultra-Fast Lane Detection (UFLD) - TuSimple ResNet18 Backbone
Paper: "Ultra Fast Structure-aware Deep Lane Detection" (ECCV 2020)
Weights: Pretrained TuSimple Benchmark (tusimple_18.pth)
"""

import os
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


class ParsingNet(nn.Module):
    """
    Ultra-Fast Lane Detection parsing network with ResNet-18 backbone.
    Outputs lane griding classification tensors across specified row anchors.
    """

    def __init__(
        self,
        size: Tuple[int, int] = (config.UFLD_INPUT_HEIGHT, config.UFLD_INPUT_WIDTH),
        cls_dim: Tuple[int, int, int] = (
            config.UFLD_NUM_GRIDDING + 1,
            config.UFLD_NUM_CLS_PER_LANE,
            config.UFLD_NUM_LANES,
        ),
    ):
        super().__init__()
        self.size = size
        self.cls_dim = cls_dim
        self.total_dim = int(np.prod(cls_dim))

        resnet = models.resnet18(weights=None)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.cls = nn.Sequential(
            nn.Linear(1800, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.total_dim),
        )
        self.pool = nn.Conv2d(512, 8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        fea = self.layer4(x3)

        fea = self.pool(fea).view(-1, 1800)
        group_cls = self.cls(fea).view(-1, *self.cls_dim)
        return group_cls


class UltraFastLaneDetector:
    """
    Ultra-Fast Lane Detection inference pipeline.
    Preprocesses input video frames, runs the deep neural network,
    decodes griding anchors to coordinate points, and projects lanes
    onto the original frame dimensions.
    """

    def __init__(
        self,
        model_path: str = config.UFLD_MODEL_PATH,
        device: str = config.DEVICE,
        min_lane_conf: float = 0.35,
        min_lane_points: int = 5,
    ):
        self.model_path = model_path
        self.device = device
        self.row_anchors = config.UFLD_ROW_ANCHORS
        self.num_gridding = config.UFLD_NUM_GRIDDING
        self.lane_colors = config.LANE_COLORS
        self.min_lane_conf = min_lane_conf
        self.min_lane_points = min_lane_points
        self.last_corridor_established = False

        self.model: Optional[ParsingNet] = None
        self.is_loaded = False

        self._load_model()

    def _load_model(self):
        """Loads pretrained weights into ParsingNet."""
        if not os.path.exists(self.model_path):
            print(f"[Module C - Warning] Pretrained weights not found at {self.model_path}.")
            print("Lane detection module will return empty detections as documented fallback.")
            self.is_loaded = False
            return

        try:
            print(f"[Module C] Loading Ultra-Fast Lane Detection model from {self.model_path} on {self.device}")
            net = ParsingNet()
            checkpoint = torch.load(self.model_path, map_location=self.device)

            if isinstance(checkpoint, dict) and "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint

            # Strip possible 'model.' namespace prefix
            cleaned_state = {}
            for k, v in state_dict.items():
                new_key = k[6:] if k.startswith("model.") else k
                cleaned_state[new_key] = v

            net.load_state_dict(cleaned_state, strict=False)
            net.to(self.device)
            net.eval()
            self.model = net
            self.is_loaded = True
            print("[Module C] Pretrained UFLD model loaded successfully!")
        except Exception as e:
            print(f"[Module C - Error] Failed to load UFLD model: {e}")
            self.is_loaded = False

    def detect(
        self,
        frame_bgr: np.ndarray,
        obstacle_boxes: Optional[List[List[float]]] = None,
    ) -> Tuple[List[List[Tuple[int, int]]], float]:
        """
        Detects road lanes in a video frame or image.
        Filters anchor points using existence confidence and suppresses lane
        points occluded by detected vehicle bounding boxes.
        Returns (lanes_points_list, inference_time_ms).
        """
        if not self.is_loaded or self.model is None:
            return [[], [], [], []], 0.0

        t0 = time.perf_counter()
        h_orig, w_orig = frame_bgr.shape[:2]

        # 1. Preprocessing: resize to (800, 288), RGB conversion, ImageNet normalization
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (config.UFLD_INPUT_WIDTH, config.UFLD_INPUT_HEIGHT))
        tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        tensor = tensor.unsqueeze(0).to(self.device)

        # 2. Forward pass
        with torch.no_grad():
            out = self.model(tensor)  # Shape: (1, 101, 56, 4)

        # 3. Postprocessing & coordinate decoding with existence confidence filtering
        prob_all = torch.softmax(out[0], dim=0)              # (101, 56, 4)
        prob_grid = torch.softmax(out[0, :-1, :, :], dim=0)  # (100, 56, 4)
        idx = torch.arange(self.num_gridding, device=self.device).float().view(self.num_gridding, 1, 1)
        loc = torch.sum(prob_grid * idx, dim=0)              # (56, 4)
        argmax = torch.argmax(out[0], dim=0)                 # (56, 4)

        lanes: List[List[Tuple[int, int]]] = []
        for lane_idx in range(config.UFLD_NUM_LANES):
            raw_pts: List[Tuple[int, int]] = []
            for r_idx in range(len(self.row_anchors)):
                # If argmax is 100, model predicts no lane at this anchor
                if argmax[r_idx, lane_idx] < self.num_gridding:
                    p_exist = 1.0 - prob_all[100, r_idx, lane_idx].item()
                    # Filter low-confidence anchors
                    if p_exist < self.min_lane_conf:
                        continue

                    col = loc[r_idx, lane_idx].item()
                    x = int(col * w_orig / float(self.num_gridding))
                    y = int(self.row_anchors[r_idx] * h_orig / float(config.UFLD_INPUT_HEIGHT))

                    # Strictly clip within image bounds
                    if 0 <= x < w_orig and 0 <= y < h_orig:
                        # Obstacle / vehicle occlusion check:
                        # Road markings cannot physically exist on top of a vehicle body
                        is_occluded = False
                        if obstacle_boxes:
                            for ob in obstacle_boxes:
                                if ob[0] <= x <= ob[2] and ob[1] <= y <= ob[3]:
                                    is_occluded = True
                                    break
                        if not is_occluded:
                            raw_pts.append((x, y))

            # Filter lateral outliers / teleportation jumps (> 25% image width)
            filtered_pts: List[Tuple[int, int]] = []
            for pt in raw_pts:
                if not filtered_pts:
                    filtered_pts.append(pt)
                else:
                    prev_x = filtered_pts[-1][0]
                    if abs(pt[0] - prev_x) < (0.25 * w_orig):
                        filtered_pts.append(pt)

            # Minimum points threshold: reject sporadic fragments
            if len(filtered_pts) >= self.min_lane_points:
                lanes.append(filtered_pts)
            else:
                lanes.append([])

        inference_time_ms = (time.perf_counter() - t0) * 1000.0
        return lanes, inference_time_ms

    def annotate_frame(
        self,
        frame_bgr: np.ndarray,
        lanes: List[List[Tuple[int, int]]],
        draw_drive_area: bool = True,
    ) -> np.ndarray:
        """
        Draws lane lines, keypoint circles, and bounded ego-lane drive-area
        polygon overlay on the original frame without altering its dimensions.
        """
        annotated = frame_bgr.copy()
        h, w = annotated.shape[:2]
        self.last_corridor_established = False

        # 1. Draw drive area polygon between Ego Left (Lane 1) and Ego Right (Lane 2)
        if draw_drive_area and len(lanes) >= 3:
            ego_left = lanes[1]
            ego_right = lanes[2]
            # Require at least min_lane_points on both ego boundaries
            if len(ego_left) >= self.min_lane_points and len(ego_right) >= self.min_lane_points:
                # Find common vertical overlap where BOTH boundaries exist
                y_min = max(ego_left[0][1], ego_right[0][1])
                y_max = min(ego_left[-1][1], ego_right[-1][1])

                # Require at least 60 px of meaningful vertical overlap
                if (y_max - y_min) >= 60:
                    pts_left = [pt for pt in ego_left if y_min <= pt[1] <= y_max]
                    pts_right = [pt for pt in ego_right if y_min <= pt[1] <= y_max]

                    if len(pts_left) >= 4 and len(pts_right) >= 4:
                        # Verify geometric ordering: left boundary must remain to the left of right boundary
                        x_left_start, x_right_start = pts_left[0][0], pts_right[0][0]
                        x_left_end, x_right_end = pts_left[-1][0], pts_right[-1][0]
                        if x_left_start < x_right_start and x_left_end < x_right_end:
                            overlay = annotated.copy()
                            poly_pts = list(pts_left) + list(reversed(pts_right))
                            poly_np = np.array(poly_pts, dtype=np.int32)
                            # Fill transparent green lane corridor
                            cv2.fillPoly(overlay, [poly_np], (0, 180, 0))
                            cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)
                            self.last_corridor_established = True

        # 2. Draw lane curve lines and node points for active lanes
        for l_idx, pts in enumerate(lanes):
            if len(pts) < 2:
                continue
            color = self.lane_colors[l_idx % len(self.lane_colors)]
            pts_arr = np.array(pts, dtype=np.int32)

            # Draw smooth polyline
            cv2.polylines(annotated, [pts_arr], isClosed=False, color=color, thickness=3, lineType=cv2.LINE_AA)

            # Draw lane keypoints
            for pt in pts:
                cv2.circle(annotated, pt, 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)

        return annotated
