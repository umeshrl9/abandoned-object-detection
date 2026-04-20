# Abandoned Object Detection Pipeline

This repository contains an **Abandoned Object Detection** system developed by **Group 327 (Umesh Kashyap – 2023UCS1693 and Yash Aggarwal – 2023UCS1698)**. The system detects unattended objects in surveillance videos using **dual background modeling** and **YOLO-based person detection**.

Unlike fully deep learning–based pipelines, this approach **avoids deep learning for background subtraction** and uses **YOLO only for human detection**, making it lightweight and suitable for real-time deployment.

---

## Overview

The proposed method combines:
- Dual background subtraction (short-term + long-term)
- Longest contour selection
- Hu invariant shape matching
- YOLO-based person detection
- Spatial human-object association
- Temporal abandonment reasoning
- Illumination change adaptation

---

## Pipeline Overview

Video Input → Dual Background Modeling → Foreground Extraction  
→ Longest Contour Selection → Shape Matching (Hu Invariants)  
→ YOLO Person Detection → Human-Object Distance Check  
→ Temporal Tracking → Abandonment Decision  

---

## Features

- YOLO-based human detection (no object detection)
- Dual background subtraction (short-term & long-term)
- Longest contour methodology
- Illumination change adaptation
- Hu invariant shape matching (`matchShapes`)
- Spatial owner association
- Temporal abandonment detection
- Zero false-positive design
- Real-time visualization

---

## Methodology

1. Read surveillance video frame-by-frame  
2. Generate short-term and long-term background models  
3. Compute difference foreground mask  
4. Apply morphological filtering  
5. Extract contours from foreground  
6. Select **longest contour** as candidate object  
7. Validate shape stability using **Hu invariants**  
8. Detect humans using **YOLO**  
9. Compute human-object distance  
10. Start timer when owner leaves  
11. If unattended beyond threshold → abandoned object  

---

## Illumination Change Handling

The system dynamically adjusts background learning rates when sudden illumination change is detected. This prevents background corruption and ensures stable foreground extraction.

---

## Example Results

### Foreground Extraction
![Foreground](images/frame.jpeg)

### Longest Contour Selection
![Longest Contour](images/long_frame.jpeg)

### Difference Frame
![Difference Frame](images/df.jpeg)

---

### Illumination Change Adaptation

Without adaptation  
![Without Illumination](images/without_illum.png)

With adaptation  
![With Illumination](images/with_illum.png)

---

### YOLO Human Detection
![Persons](images/persons.png)

---

### Final Abandoned Object Detection
![Result](images/video_7.png)

---

## Parameters

| Parameter | Value |
|-----------|-------|
| Abandonment time | 5 seconds |
| Minimum contour area | 500 |
| Short-term learning rate | 0.05 |
| Long-term learning rate | 0.00025 |
| Fast learning rate (illumination) | 0.5 |
| Illumination threshold | 0.3 |
| Shape matching | Hu invariants |
| Human detection | YOLO |
| Dataset | ABODA |

---

## Techniques Used

- Dual Background Modeling  
- Longest Contour Selection  
- Morphological Filtering  
- Hu Invariant Shape Matching  
- YOLO Human Detection  
- Temporal Tracking  
- Spatial Distance Thresholding  
- Illumination Adaptation  

---

## Advantages

- Avoids heavy deep-learning background subtraction
- More efficient than fully DL-based approaches (e.g., Kim et al.)
- More robust human detection than HOG-based approaches (e.g., Park et al.)
- Works under illumination changes
- Zero false positives on evaluated videos
- Real-time capable

---

## Dataset

Evaluation performed on **ABODA Dataset** including:
- Outdoor scenes
- Indoor scenes
- Night scenarios
- Illumination changes
- Crowded scenes

---

## How to Run

```bash
git clone https://github.com/umeshrl9/abandoned-object-detection.git
cd abandoned-object-detection
pip install opencv-python numpy ultralytics
python main.py
