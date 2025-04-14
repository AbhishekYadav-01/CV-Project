import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def extract_object_mask(scene, object_scene):
    """
    Extracts the mask of the object by subtracting the scene from object_scene.
    """
    diff = cv2.absdiff(scene, object_scene)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)

    # Refine mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def estimate_shadow_direction(scene):
    """
    Estimate shadow direction based on edge detection and dominant angles.
    """
    gray = cv2.cvtColor(scene, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    if lines is None:
        return (20, 20)  # fallback direction

    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = np.degrees(theta)
        if 20 < angle < 160:
            angles.append(angle)
    if not angles:
        return (20, 20)
    dominant_angle = np.median(angles)
    rad = np.radians(dominant_angle)
    dx = int(50 * np.cos(rad))
    dy = int(50 * np.sin(rad))
    return dx, dy

def project_shadow(object_mask, direction, blur=5):
    """
    Project a soft shadow based on the object mask and direction.
    """
    dx, dy = direction
    h, w = object_mask.shape
    shadow_mask = np.roll(object_mask, shift=(dy, dx), axis=(0, 1))
    shadow_mask = np.clip(shadow_mask, 0, 255)
    shadow = gaussian_filter(shadow_mask.astype(float), sigma=blur)
    shadow = (shadow / shadow.max() * 120).astype(np.uint8)
    return shadow

def blend_object_with_shadow(scene, object_scene, object_mask, shadow_mask):
    """
    Blend the object and its shadow into the scene.
    """
    shadow_colored = cv2.merge([shadow_mask] * 3)
    darkened = (scene * 0.6).astype(np.uint8)
    shadowed_scene = np.where(shadow_colored > 30, darkened, scene)

    object_mask_3 = cv2.merge([object_mask] * 3)
    final = np.where(object_mask_3 > 0, object_scene, shadowed_scene)
    return final

def apply_realistic_compositing(scene_path, object_scene_path, output_path='final_output.jpg'):
    """
    Full pipeline for realistic compositing with lighting and shadows.
    """
    # Load images
    scene = cv2.imread(scene_path)
    object_scene = cv2.imread(object_scene_path)
    if scene is None or object_scene is None:
        raise FileNotFoundError("Check image paths.")
    scene = cv2.cvtColor(scene, cv2.COLOR_BGR2RGB)
    object_scene = cv2.cvtColor(object_scene, cv2.COLOR_BGR2RGB)

    # Extract object mask
    object_mask = extract_object_mask(scene, object_scene)

    # Estimate shadow direction
    shadow_dir = estimate_shadow_direction(scene)

    # Project shadow
    shadow_mask = project_shadow(object_mask, shadow_dir)

    # Blend object and shadow into the scene
    final_img = blend_object_with_shadow(scene, object_scene, object_mask, shadow_mask)

    # Save the final image
    final_img_bgr = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, final_img_bgr)
    print(f"✅ Final image saved to: {output_path}")

    # Display the result
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title("Original Scene")
    plt.imshow(scene)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("With Object")
    plt.imshow(object_scene)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Final Composite")
    plt.imshow(final_img)
    plt.axis("off")
    plt.show()
