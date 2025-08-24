# 🛣️ Pothole Preventer – Smart Path Planning with YOLO Segmentation

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Segmentation-orange?logo=ultralytics)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 📌 Overview

**Pothole Preventer** is a computer vision project that detects road damage in real-time and suggests the **smoothest and safest path** for a vehicle to follow.  
It uses **YOLOv8 segmentation** to detect potholes and road damage, overlays a **heatmap of road conditions**, and computes an **optimal driving path** by minimizing cumulative damage along the route.

The system works with both **pre-recorded videos** and **live camera feeds**.

---

## 🚀 Features

- ✅ **YOLOv8 segmentation** for accurate pothole/damage detection
- ✅ **Real-time heatmap generation** showing road quality
- ✅ **Dynamic path planning** (minimizes total damage)
- ✅ **Smoothed path rendering** (no jerky swerving, suitable for real cars)
- ✅ Works with both **video files** and **live camera input**

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/CodeThread386/Pothole-Preventer.git
cd Pothole-Preventer
```

### 2. Install Dependencies

Install the required Python packages:

```bash
pip install ultralytics opencv-python numpy matplotlib
```

---

## ▶️ Usage

### Run on a Video File

```bash
python main.py --input path/to/video.mp4
```

### Run on Live Camera

```bash
python main.py --live
```

---

## 📊 How It Works

1. **Segmentation** – YOLOv8 detects potholes/damage and masks them.
2. **Tile Grid Overlay** – The road is divided into tiles, each scored by damage percentage.
3. **Path Planning** – At each frame, the algorithm chooses the next tile to minimize cumulative damage.
4. **Path Smoothing** – The path is filtered to remove jerky swerves, producing a realistic drivable line.
5. **Visualization** – Heatmap and the recommended driving path (yellow) are overlaid on the video.

---

## 🔮 Future Improvements

- 📍 GPS integration for real-world navigation
- 🚗 Lane-aware planning
- 🧠 Adaptive smoothing based on vehicle dynamics
- 📱 Mobile app deployment (dashcam integration)
