import cv2
import numpy as np
from typing import Optional, Tuple

def detect_lines(image):
    """
    Detect lines in the image using the probabilistic Hough Transform.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)
    return lines

def compute_intersections(lines):
    """
    Compute all intersections between detected lines.
    """
    intersections = []
    if lines is None:
        return intersections
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            l1, l2 = lines[i][0], lines[j][0]
            denom = (l1[0] - l1[2]) * (l2[1] - l2[3]) - (l1[1] - l1[3]) * (l2[0] - l2[2])
            if denom != 0:
                Px = ((l1[0] * l1[3] - l1[1] * l1[2]) * (l2[0] - l2[2]) - (l1[0] - l1[2]) * (l2[0] * l2[3] - l2[1] * l2[2])) / denom
                Py = ((l1[0] * l1[3] - l1[1] * l1[2]) * (l2[1] - l2[3]) - (l1[1] - l1[3]) * (l2[0] * l2[3] - l2[1] * l2[2])) / denom
                intersections.append((Px, Py))
    return intersections

def cluster_vanishing_points(intersections):
    """
    Cluster the set of intersection points to determine dominant vanishing points.
    """
    if len(intersections) == 0:
        return None
    intersections_np = np.array(intersections)
    _, _, centers = cv2.kmeans(
        intersections_np.astype(np.float32), 2, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS
    )
    return centers

def estimate_camera_pose(image: np.ndarray, depth: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate camera intrinsics and extrinsics based on vanishing points.
    """
    h, w = image.shape[:2]

    # Heuristic focal length estimation
    f = 1.2 * w
    cx, cy = w / 2, h / 2
    camera_matrix = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

    # Detect lines in the image
    lines = detect_lines(image)
    # Compute intersections of these lines
    intersections = compute_intersections(lines)
    # Cluster intersections to obtain vanishing point candidates
    vanishing_points = cluster_vanishing_points(intersections)

    # Estimate a simple rotation matrix from vanishing points
    if vanishing_points is not None and len(vanishing_points) >= 2:
        vp1, vp2 = vanishing_points[0], vanishing_points[1]
        dir1 = np.array([vp1[0] - cx, vp1[1] - cy, f])
        dir2 = np.array([vp2[0] - cx, vp2[1] - cy, f])
        dir1_norm = dir1 / np.linalg.norm(dir1)
        dir2_norm = dir2 / np.linalg.norm(dir2)
        z_axis = np.cross(dir1_norm, dir2_norm)
        z_axis = z_axis / np.linalg.norm(z_axis)
        x_axis = np.cross(dir2_norm, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        R = np.stack([x_axis, dir2_norm, z_axis], axis=1)
    else:
        # Fallback: Identity rotation if vanishing points are insufficient
        R = np.eye(3)

    # Integrating Depth Information for Translation
    if depth is not None:
        bottom_region = depth[int(h * 0.9):, :]
        median_depth = np.median(bottom_region)
        t = np.array([[0], [0], [median_depth]])
    else:
        t = np.zeros((3, 1))

    # Assemble the final 4x4 pose matrix
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = R
    pose_matrix[:3, 3] = t.flatten()

    return camera_matrix, pose_matrix
