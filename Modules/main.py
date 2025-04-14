from download_assets import download_assets
from scene_geometry import load_midas_model, estimate_scene_geometry
from camera_pose import estimate_camera_pose
from object_placement import place_object
from lighting_adaptation import apply_realistic_compositing
import cv2
import numpy as np

def main():
    # Step 1: Download assets
    download_assets()

    # Step 2: Estimate scene geometry
    print("Loading scene image...")
    image_path = "scene_image.jpeg"
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Scene image not found. Please check the path.")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print("Loading MiDaS model...")
    midas, transform, device = load_midas_model()
    print("Estimating scene geometry...")
    depth_map, normal_map = estimate_scene_geometry(img_rgb, midas, transform, device)
    np.save("depth_map.npy", depth_map)
    np.save("normal_map.npy", normal_map)
    print("Depth map and normal map saved.")

    # Step 3: Estimate camera pose
    print("Estimating camera pose...")
    camera_matrix, pose_matrix = estimate_camera_pose(img, depth_map)
    np.save("pose_matrix.npy", pose_matrix)
    print("Camera pose matrix saved.")

    # Step 4: Place object
    print("Placing object in the scene...")
    object_model_path = "object.glb"
    placement_point = (img.shape[1] // 2, int(0.8 * img.shape[0]))  # Center-bottom placement
    object_scale = 0.1
    object_offset = -0.7
    angle_x, angle_y, angle_z = 0.0, 0.0, 0.0  # Default angles
    camera_params = (camera_matrix, pose_matrix)
    result_img, _ = place_object(
        img_rgb, object_model_path, camera_params, depth_map, placement_point,
        object_scale, object_offset, angle_x, angle_y, angle_z
    )
    cv2.imwrite("scene_with_object.png", cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
    print("Object placed and saved as 'scene_with_object.png'.")

    # Step 5: Apply lighting and shadow adaptation
    print("Applying lighting and shadow adaptation...")
    apply_realistic_compositing("scene_image.jpeg", "scene_with_object.png", "final_output.jpg")
    print("Final composited image saved as 'final_output.jpg'.")

if __name__ == "__main__":
    main()
