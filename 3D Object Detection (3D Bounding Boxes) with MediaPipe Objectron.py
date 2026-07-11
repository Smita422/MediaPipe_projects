import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils

# Open Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

with mp_objectron.Objectron(
    static_image_mode=False,
    max_num_objects=5,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_name='Cup'      # Change to 'Chair', 'Shoe', or 'Camera' if needed
) as objectron:

    while True:
        success, frame = cap.read()

        if not success:
            print("Failed to read frame.")
            break

        # Flip for selfie view
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Improve performance
        rgb.flags.writeable = False
        results = objectron.process(rgb)
        rgb.flags.writeable = True

        # Draw detections
        if results.detected_objects:
            for detected_object in results.detected_objects:

                # Draw 2D bounding box landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    detected_object.landmarks_2d,
                    mp_objectron.BOX_CONNECTIONS
                )

                # Draw 3D coordinate axes
                mp_drawing.draw_axis(
                    frame,
                    detected_object.rotation,
                    detected_object.translation
                )

        cv2.imshow("MediaPipe Objectron - Real Time", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()