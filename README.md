# 🧠 3D Object Insertion into 2D Scenes: A Multi-Module Augmented Reality Pipeline

This project demonstrates a modular pipeline for inserting 3D objects into 2D scene images with high realism. The system ensures the correct perspective, occlusion, lighting, and interactivity to deliver a complete AR experience — all using a combination of classical computer vision and deep learning.

---

## 📌 Project Overview

The pipeline consists of five core modules:

1. **Scene Geometry Estimation** – Extracts depth and surface normals from a 2D image using MiDaS.
2. **Camera Pose Estimation** – Computes camera orientation and position from vanishing points and optional depth.
3. **Object Placement and Occlusion Handling** – Places the 3D object in the correct position with occlusion-aware blending.
4. **Lighting and Shadow Adaptation** – Matches lighting and shadows with the scene using classical and GAN-based methods.
5. **Deployment and Interactivity** – A Streamlit interface allows users to upload scenes and insert 3D objects interactively.

---

## 📷 Demo Highlights

- 3D Spider-Man placed realistically into real-world scenes.
- Camera pose visualized with 3D axis projections.
- Occlusion masks generated from depth data.
- Shadows adapted using multiple methods including ShadowGAN.

---

## 🛠️ Technologies Used

- **Python**
- **OpenCV**, **NumPy**
- **MiDaS (DPT-Large)** for depth estimation
- **Trimesh**, **Pyrender** for 3D mesh and rendering
- **Scikit-learn (K-means)** for vanishing point clustering
- **Streamlit** for UI deployment
- **ShadowGAN** for advanced shadow rendering

---

## 📂 Directory Structure
```bash
├── modules/
│   ├── depth_estimation.py             # Depth and normal map computation using MiDaS
│   ├── camera_pose.py                  # Line detection, vanishing points, and pose matrix
│   ├── object_placement.py             # Object transformation, rendering, and occlusion
│   ├── lighting_shadow.py              # Lighting correction and shadow generation methods
│   └── ui_streamlit.py                 # Streamlit-based user interface
│
├── models/
│   └── object.glb                      # 3D object model used in rendering
│
├── data/
│   ├── scene_image.jpeg                # Input 2D scene image
│   └── depth_map.npy                   # Predicted depth map (optional)
│
├── outputs/
│   ├── final_composite.png            # Final image with the inserted 3D object
│   ├── pose_matrix.npy                # Estimated 4x4 camera pose matrix
│   └── debug_visuals/                 # Optional visualizations (axes, masks, etc.)
│
├── app.py                              # Main script to run the entire pipeline
├── README.md                           # Project documentation
└── requirements.txt                    # Python dependencies
```


---

## 🧑‍💻 Team Members and Contributions

- **Omprakash Nain (B22AI062)** – Scene Geometry Estimation
- **Shahil Sharma (B22CS048)** – Camera Pose Estimation via Geometric Reasoning
- **Abhishek Yadav (B22ES020)** – Object Placement and Occlusion Handling
- **Sai Vignesh (B22ES023)** – Model Loading, Positioning, ShadowGAN Integration
- **Shivanshu Verma (B22ES010)** – Lighting, Shadow Adaptation & UI Deployment

---

## 📈 Results

- Composite images show realistic placement and lighting of virtual 3D objects.
- Depth- and normal-based occlusion blending makes object integration seamless.
- Multiple shadow techniques compared to find the best visual quality.

---

## 🔮 Future Work

- Real-time video integration
- Improved pose estimation with learning-based vanishing point detection
- Mobile deployment using lightweight inference

---

## 🔗 GitHub Repository

[Click here to view the code on GitHub](https://github.com/AbhishekYadav-01/CV-Project)

---

## 📅 Date of Submission

**April 14, 2025**
