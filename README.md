# Abandoned Object Detection Pipeline

This repository contains Abandoned Object Detection project developed by **Group 27 (Umesh Kashyap, 2023UCS1693 and Yash Aggarwal, 2023UCS1698)** that detects abandoned objects in surveillance videos.  
The system uses **YOLO11n for pedestrian detection** and combines it with **classical computer vision, temporal tracking, and spatial analysis** to identify unattended objects.

Unlike fully deep learning-based approaches, this pipeline relies on **mathematical logic and background modeling**, using deep learning only for detecting humans.

---

## Features

- YOLO11n-based pedestrian detection  
- Dual background subtraction (short-term & long-term)  
- Illumination change handling  
- Candidate object stability tracking  
- Spatial human-object association  
- Temporal abandonment detection  
- Shape matching verification  
- Real-time visualization  

---

##  Methodology

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

##  Techniques Used

- YOLO11n (Human Detection Only)  
- Background Subtraction (KNN)  
- Morphological Operations  
- Centroid Distance Calculation  
- Temporal Thresholding  
- Contour Shape Matching  
- Illumination Adaptation Logic  

---

##  How to Run

```bash
git clone https://github.com/umeshrl9/abandoned-object-detection.git
cd abandoned-object-detection
pip install cv2 numpy time ultralytics
python main.py
