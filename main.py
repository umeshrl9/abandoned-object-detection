import cv2
import numpy as np
import time
from ultralytics import YOLO

VIDEO_PATH = "Dataset/ABODA/video1.avi" #Path of video to be tested
ABANDON_TIME = 5  
MIN_AREA = 500

LR_SHORT_NORMAL = 0.05
LR_LONG_NORMAL  = 0.0001
LR_FAST = 0.5

ILLUMINATION_THRESHOLD = 0.3
OWNER_DIST_THRESHOLD = 300   

# YOLO11n
model = YOLO("yolo11n.pt")

def detect_humans(frame):
    results = model(frame, verbose=False)[0]
    boxes = []
    for box in results.boxes:
        if int(box.cls[0]) == 0: 
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2))
    return boxes

# OWNER CHECK
def is_owner_near(bbox, human_boxes):
    x, y, w, h = bbox
    cx, cy = x + w//2, y + h//2

    for (hx1, hy1, hx2, hy2) in human_boxes:
        hx = (hx1 + hx2)//2
        hy = (hy1 + hy2)//2

        dist = ((cx - hx)**2 + (cy - hy)**2)**0.5
        if dist < OWNER_DIST_THRESHOLD:
            return True

    return False

# BACKGROUND MODELS
bg_short = cv2.createBackgroundSubtractorKNN(history=50, detectShadows=False)
bg_long  = cv2.createBackgroundSubtractorKNN(history=1000, detectShadows=False)

illumination_flag = False


# STORAGE
candidates = []

# MAIN LOOP
cap = cv2.VideoCapture(VIDEO_PATH)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Learning Rate Control
    if illumination_flag:
        lr_s = LR_FAST
        lr_l = LR_FAST
    else:
        lr_s = LR_SHORT_NORMAL
        lr_l = LR_LONG_NORMAL

    fg_short = bg_short.apply(frame, learningRate=lr_s)
    fg_long  = bg_long.apply(frame, learningRate=lr_l)

    df = cv2.subtract(fg_long, fg_short)

    _, df = cv2.threshold(df, 200, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3,3), np.uint8)

    # remove small noise
    df = cv2.morphologyEx(df, cv2.MORPH_OPEN, kernel, iterations=2)

    # fill holes
    df = cv2.morphologyEx(df, cv2.MORPH_CLOSE, kernel, iterations=2)

    # merge nearby pixels
    df = cv2.dilate(df, kernel, iterations=1)

    # Illumination Detection
    contours_lf, _ = cv2.findContours(fg_long, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = max([cv2.contourArea(c) for c in contours_lf], default=0)
    frame_area = frame.shape[0] * frame.shape[1]

    if max_area > ILLUMINATION_THRESHOLD * frame_area:
        illumination_flag = True

    # Stop adaptation
    contours_df, _ = cv2.findContours(df, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if illumination_flag and len(contours_df) == 0:
        illumination_flag = False

    # Human Detection
    human_boxes = detect_humans(frame)

    # Candidate Detection
    for cnt in contours_df:
        if cv2.contourArea(cnt) < MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        matched = False

        for obj in candidates:
            ox, oy, ow, oh = obj["bbox"]
            area = w*h
            oarea = ow*oh

            if (abs(x - ox) < 30 and abs(y - oy) < 30 and abs(area - oarea) < 500):
                obj["frames"] += 1
                matched = True

                # Stability achieved
                if obj["frames"] == 30:
                    obj["template"] = cnt
                    obj["stable"] = True

                break

        if not matched:
            candidates.append({
                "bbox": (x, y, w, h),
                "frames": 1,
                "template": None,
                "stable": False,
                "timer_started": False,
                "start_time": None
            })

    # Temporal + Spatial Logic
    for obj in candidates:
        if not obj["stable"]:
            continue

        x, y, w, h = obj["bbox"]

        # Check if owner is near
        if is_owner_near((x,y,w,h), human_boxes):
            obj["timer_started"] = False
            obj["start_time"] = None
            continue

        # Owner left → start timer
        if not obj["timer_started"]:
            obj["start_time"] = time.time()
            obj["timer_started"] = True

        # Check abandonment
        if obj["timer_started"]:
            elapsed = time.time() - obj["start_time"]

            if elapsed > ABANDON_TIME:
                roi = df[y:y+h, x:x+w]
                contours2, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if len(contours2) == 0:
                    continue

                largest = max(contours2, key=cv2.contourArea)

                score = cv2.matchShapes(obj["template"], largest, 1, 0.0)

                if score < 0.2:
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
                    cv2.putText(frame, "ABANDONED", (x,y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # Draw Humans
    for (x1,y1,x2,y2) in human_boxes:
        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)

    # Display
    cv2.imshow("Frame", frame)
    cv2.imshow("DF", df)
    cv2.imshow("FG Long", fg_long)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
