import cv2  # OpenCV – library for image and video processing
from ultralytics import YOLO  # Ultralytics YOLO – object detection model

# Prompt user to enter the IP address and port
ip = input("Enter the camera IP address (e.g., 192.168.1.100): ").strip()
port = input("Enter the port (default is 4747): ").strip()
if not port:
    port = "4747"

# Build the MJPEG stream URL from DroidCam
STREAM_URL = f"http://{ip}:{port}/video"

# Load pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")  # nano version – fast and lightweight

# Open the video stream from phone or webcam
cap = cv2.VideoCapture(STREAM_URL)

# Frame counter for skipping
frame_count = 0

while True:
    # Read a single video frame
    ret, frame = cap.read()

    # If reading failed, exit loop
    if not ret:
        print("No image. Check camera connection.")
        break

    # Increase frame counter
    frame_count += 1

    # Skip every other frame (process every 2nd frame)
    if frame_count % 3 != 0:
        continue

    # Run YOLO detection on the frame
    results = model(frame, verbose=False)

    # Draw bounding boxes only for detected persons
    for box in results[0].boxes:
        cls = int(box.cls[0])  # Get class ID
        if cls == 0:  # 0 corresponds to "person" in the COCO dataset
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
            cv2.putText(
                frame,
                "Person",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    # Show the frame with detected people
    cv2.imshow("Detect People", frame)

    # Exit loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video stream and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
