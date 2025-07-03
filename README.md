# Human-Movement-Detection
import cv2

# === Load pre-recorded video ===
video_path = "video.mp4"
cap = cv2.VideoCapture(video_path)

# === Check if video loaded properly ===
if not cap.isOpened():
    print(f" Error: Couldn't open video file '{video_path}'")
    exit()

# === Print video info ===
print("🎥 Video Properties:")
print(" - Frame Width:  ", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(" - Frame Height: ", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(" - FPS:          ", cap.get(cv2.CAP_PROP_FPS))
print(" - Total Frames: ", cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("====================================")

# === Initialize background subtractor ===
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100)

# === Process video ===
while True:
    ret, frame = cap.read()
    if not ret:
        print(" Video ended or couldn't read frame.")
        break

    # Resize for speed (optional)
    frame = cv2.resize(frame, (640, 480))

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Find contours from the mask
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around motion
    for contour in contours:
        if cv2.contourArea(contour) < 800:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the frame
    cv2.imshow(" Motion Detection - Press 'q' to quit", frame)

    # Exit on 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        print(" Exiting on user command.")
        break

# === Clean up ===
cap.release()
cv2.destroyAllWindows()
