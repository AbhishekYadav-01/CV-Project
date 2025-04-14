import torch
import cv2
import numpy as np
from typing import Tuple

def load_midas_model(model_type="DPT_Large"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform if model_type in ["DPT_Large", "DPT_Hybrid"] else midas_transforms.small_transform

    return midas, transform, device

def estimate_scene_geometry(image: np.ndarray, midas, transform, device) -> Tuple[np.ndarray, np.ndarray]:
    input_batch = transform(image).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()
    depth_map = prediction.cpu().numpy()

    depth_min, depth_max = depth_map.min(), depth_map.max()
    depth_map_norm = (depth_map - depth_min) / (depth_max - depth_min) if depth_max - depth_min > 1e-6 else np.zeros(depth_map.shape, dtype=np.float32)

    grad_x = cv2.Sobel(depth_map_norm, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(depth_map_norm, cv2.CV_64F, 0, 1, ksize=5)
    normal_map = np.dstack((-grad_x, -grad_y, np.ones_like(depth_map_norm)))
    norm = np.linalg.norm(normal_map, axis=2, keepdims=True)
    normal_map = normal_map / np.maximum(norm, 1e-6)

    return depth_map_norm, normal_map
