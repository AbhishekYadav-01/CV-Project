import trimesh
import pyrender
import numpy as np
import cv2
from typing import Tuple, Optional

def compose_rotation(angle_x: float, angle_y: float, angle_z: float) -> np.ndarray:
    rx, ry, rz = np.deg2rad(angle_x), np.deg2rad(angle_y), np.deg2rad(angle_z)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def place_object(scene: np.ndarray, object_model_path: str, camera_params: Tuple[np.ndarray, np.ndarray], 
                 scene_depth: Optional[np.ndarray] = None, placement_point: Optional[Tuple[int, int]] = None, 
                 object_scale: float = 1.0, object_offset: float = 0.0, angle_x: float = 0.0, angle_y: float = 0.0, 
                 angle_z: float = 0.0) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Place a 3D object into the scene with proper scaling, rotation, and occlusion handling.

    Parameters:
        scene (np.ndarray): Input scene image.
        object_model_path (str): Path to the 3D object model file.
        camera_params (Tuple[np.ndarray, np.ndarray]): Camera intrinsic and extrinsic matrices.
        scene_depth (Optional[np.ndarray]): Depth map of the scene.
        placement_point (Optional[Tuple[int, int]]): Pixel coordinates for object placement.
        object_scale (float): Scaling factor for the object.
        object_offset (float): Offset along the Z-axis for object placement.
        angle_x (float): Rotation angle around the X-axis.
        angle_y (float): Rotation angle around the Y-axis.
        angle_z (float): Rotation angle around the Z-axis.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: Rendered scene with the object and optional occlusion mask.
    """
    camera_matrix, pose_matrix = camera_params
    h, w, _ = scene.shape

    if placement_point is None:
        placement_point = (w // 2, h // 2)
    click_x, click_y = placement_point
    print(f"Placement point (image coordinates): {placement_point}")

    try:
        mesh = trimesh.load(object_model_path)
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) == 0:
                raise RuntimeError("The scene contains no geometry.")
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        print(f"Loaded mesh with {len(mesh.vertices)} vertices.")
        mesh.apply_scale(object_scale)
        print(f"Applied object scale: {object_scale}")
    except Exception as e:
        raise RuntimeError(f"Failed to load object model: {e}")

    render_scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.1, 0.1, 0.1])
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=10.0)
    render_scene.add(light, pose=np.eye(4))
    render_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)

    if scene_depth is None:
        depth_val = 1.0
    else:
        depth_val = scene_depth[min(click_y, scene_depth.shape[0] - 1),
                                min(click_x, scene_depth.shape[1] - 1)]
        if depth_val <= 0:
            print("Depth value non-positive. Using fallback 1.5.")
            depth_val = 1.5
    print(f"Depth at placement point: {depth_val}")

    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    new_depth = depth_val + object_offset
    X = (click_x - cx) * new_depth / fx
    Y = (click_y - cy) * new_depth / fy
    Z = new_depth
    placement_3d = np.array([X, Y, Z, 1.0]).reshape((4, 1))
    print(f"Camera coordinate placement (homogeneous): {placement_3d.ravel()}")

    world_placement = np.linalg.inv(pose_matrix) @ placement_3d
    world_placement = world_placement.flatten()[:3]
    print(f"World coordinate placement: {world_placement}")

    translation = np.eye(4)
    translation[:3, 3] = world_placement

    R_custom = compose_rotation(angle_x, angle_y, angle_z)
    object_transform = translation @ R_custom

    print("Final object transform matrix:")
    print(object_transform)

    obj_node = pyrender.Node(mesh=render_mesh, matrix=object_transform)
    render_scene.add_node(obj_node)

    camera_pose = np.linalg.inv(pose_matrix)
    camera_intrinsics = pyrender.IntrinsicsCamera(
        fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=10000.0
    )
    render_scene.add(camera_intrinsics, pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
    render_color, render_depth = renderer.render(render_scene)
    renderer.delete()
    print("Rendering complete.")

    scene_float = scene.astype(np.float32) / 255.0
    object_float = render_color.astype(np.float32) / 255.0

    non_zero = (np.sum(object_float, axis=2) > 0).astype(np.float32)[:, :, np.newaxis]
    result_img = object_float * non_zero + scene_float * (1 - non_zero)
    result_img = (result_img * 255).astype(np.uint8)

    return result_img, None
