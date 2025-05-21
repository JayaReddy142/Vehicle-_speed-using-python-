import cv2
from ultralytics import YOLO
import time

yolo = YOLO("yolov8s.pt")
cap = cv2.VideoCapture(r'C:\Users\sanga\Downloads\WhatsApp Video 2025-05-10 at 3.26.54 PM.mp4')


# Dictionary to hold object IDs and their positions/timestamps
vehicle_data = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo.track(frame, persist=True, stream=True)

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            if result.names[class_id] != "car":
                continue  # Skip if not a car (you can add 'truck', 'bus', etc.)

            # Get object ID (YOLO assigns unique ids in tracking)
            object_id = int(box.id[0]) if box.id is not None else None
            if object_id is None:
                continue

            # Get center of bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            current_time = time.time()

            # Check if we've seen this ID before
            if object_id in vehicle_data:
                old_cx, old_cy, old_time = vehicle_data[object_id]
                pixel_dist = ((cx - old_cx) ** 2 + (cy - old_cy) ** 2) ** 0.5
                time_diff = current_time - old_time

                if time_diff > 0:
                    speed_px_per_sec = pixel_dist / time_diff
                    speed_kmph = (speed_px_per_sec * 0.07)  # Convert (rough scale)
                    cv2.putText(frame, f"Speed: {speed_kmph:.2f} km/h", (x1, y1 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Update data
            vehicle_data[object_id] = (cx, cy, current_time)

            # Draw box
            label = f"{result.names[class_id]} ID: {object_id}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Vehicle Speed Estimator", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()