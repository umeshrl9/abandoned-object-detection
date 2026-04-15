# Abandoned Object Detection Pipeline

This repository contains a **college project** developed by **Group 27** that detects abandoned objects in surveillance videos.  
The system uses **YOLO11n for pedestrian detection** and combines it with **classical computer vision, temporal tracking, and spatial analysis** to identify unattended objects.

Unlike fully deep learning-based approaches, this pipeline relies on **mathematical logic and background modeling**, using deep learning only for detecting humans.

---

## 📌 Features

- YOLO11n-based pedestrian detection  
- Dual background subtraction (short-term & long-term)  
- Illumination change handling  
- Candidate object stability tracking  
- Spatial human-object association  
- Temporal abandonment detection  
- Shape matching verification  
- Real-time visualization  

---

## ⚙️ Methodology

The pipeline follows these steps:

1. Read surveillance video frame-by-frame  
2. Detect humans using YOLO11n  
3. Generate short-term and long-term background models  
4. Compute difference foreground to extract static objects  
5. Apply morphological filtering to remove noise  
6. Track candidate objects across frames  
7. Mark objects as stable after consistent detection  
8. Check if a human is near the object (owner association)  
9. Start timer when owner leaves  
10. If unattended beyond threshold → classify as abandoned  
11. Validate using contour shape matching  

---

## 🧠 Techniques Used

- YOLO11n (Human Detection Only)  
- Background Subtraction (KNN)  
- Morphological Operations  
- Centroid Distance Calculation  
- Temporal Thresholding  
- Contour Shape Matching  
- Illumination Adaptation Logic  

---

## 📊 Parameters

| Parameter | Value |
|-----------|-------|
| Abandonment Time | 5 seconds |
| Minimum Object Area | 500 pixels |
| Owner Distance Threshold | 300 pixels |
| Stability Frames | 30 frames |
| Illumination Threshold | 30% frame area |

---

## 🛠️ Tech Stack

- Python  
- OpenCV  
- NumPy  
- Ultralytics YOLO11n  

---

## ▶️ How to Run

```bash
git clone <repository-link>
cd abandoned-object-detection
pip install -r requirements.txt
python main.py