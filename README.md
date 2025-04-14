# CV-Project: Realistic Object Placement and Compositing

This project demonstrates a pipeline for placing 3D objects into 2D scenes with realistic lighting, shadows, and geometry adaptation. It combines computer vision and rendering techniques to achieve visually convincing results.

## Features

- **3D Object Placement**: Place 3D objects into 2D scenes using camera parameters and depth maps.
- **Scene Geometry Estimation**: Use MiDaS to estimate depth and normal maps for the scene.
- **Lighting and Shadow Adaptation**: Apply realistic lighting and shadow effects to blend objects seamlessly into the scene.
- **Camera Pose Estimation**: Compute camera intrinsics and extrinsics for accurate object placement.

## Project Structure

```
CV-Project/
├── Modules/
│   ├── download_assets.py          # Downloads required assets (3D object and scene image)
│   ├── scene_geometry.py           # Estimates scene geometry (depth and normal maps)
│   ├── camera_pose.py              # Estimates camera pose
│   ├── object_placement.py         # Places 3D objects into the scene
│   ├── lighting_adaptation.py      # Adapts lighting and shadows for realism
│   ├── main.py                     # Main pipeline script
├── README.md                       # Project documentation
```

## Setup Instructions

1. **Clone the Repository**:

   ```bash
   git clone <repository-url>
   cd CV-Project
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.8+ installed. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download Assets**:
   The project includes a script to download the necessary assets:

   ```bash
   python Modules/download_assets.py
   ```

4. **Run the Pipeline**:
   Execute the main script to run the full pipeline:
   ```bash
   python Modules/main.py
   ```

## Usage

### Step 1: Download Assets

The `download_assets.py` script downloads a sample 3D object (`object.glb`) and a scene image (`scene_image.jpeg`) for testing.

### Step 2: Estimate Scene Geometry

The `scene_geometry.py` module uses MiDaS to estimate depth and normal maps for the input scene image.

### Step 3: Estimate Camera Pose

The `camera_pose.py` module computes the camera's intrinsic and extrinsic parameters.

### Step 4: Place Object

The `object_placement.py` module places the 3D object into the scene using the estimated geometry and camera parameters.

### Step 5: Apply Lighting and Shadows

The `lighting_adaptation.py` module blends the object into the scene with realistic lighting and shadow effects.

### Step 6: View Results

The final composited image is saved as `final_output.jpg`.

## Example Output

- **Input Scene**: `scene_image.jpeg`
- **3D Object**: `object.glb`
- **Final Composite**: `final_output.jpg`

## Dependencies

- Python 3.8+
- OpenCV
- PyTorch
- NumPy
- PyRender
- Trimesh
- Matplotlib
- gdown

## Acknowledgments

- **MiDaS**: For depth and normal map estimation.
- **PyRender**: For 3D rendering.
- **Trimesh**: For 3D object manipulation.

## Contributions

- Om Prakash Nain : Depth and Normal Estimation
- Shahil Sharma : Pose Estimation
  = Abhishek Yadav & Vignesh : Object Placement and Occulsion
- Shivanshu Verm : Lighting and Shadow Adaptation
- Shivanshu Verma & Vignesh : Project Deployment

## License

This project is licensed under the MIT License. See the LICENSE file for details.
