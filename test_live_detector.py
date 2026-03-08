import cv2
from detector import YOLOv9Detector

detector = YOLOv9Detector(
    weights="yolov9/runs/train/fire_person/weights/best.pt",
    conf_thres=0.10
)

cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        print("Camera read failed")
        break

    detections = detector.detect(frame)
    print("Detections:", detections)

    for box, label, conf in detections:
        x1, y1, x2, y2 = box
        color = (0, 255, 0) if label == "person" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Test Detector", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()