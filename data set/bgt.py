import cv2
import numpy as np
from ultralytics import YOLO

class BGTTracker:
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        self.initialized = False
        self.initial_size = None
        self.track_window = None
        self.last_position = None

    def initialize(self, frame, bbox):
        """ Initialize the tracker with the first frame and the bounding box. """
        x, y, w, h = bbox
        self.track_window = (x, y, w, h)
        self.initial_size = (w, h)
        self.initialized = True
        self.last_position = (x + w // 2, y + h // 2)  # Center of the bounding box

    def update(self, frame):
        """ Update the tracker with the current frame and return the new position of the bounding box. """
        if not self.initialized:
            return None

        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)

        # Apply morphological operations to reduce noise and fill gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # Threshold to improve the tracking
        _, thresh = cv2.threshold(fg_mask, 50, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Sort contours by area and keep the largest
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 0 and h > 0:  # Ensure non-zero size
                    # Adjust to maintain initial size ratio
                    x_center = x + w // 2
                    y_center = y + h // 2
                    initial_w, initial_h = self.initial_size
                    
                    # Smooth the position to reduce jitter
                    if self.last_position:
                        x_center = int(0.7 * self.last_position[0] + 0.3 * x_center)
                        y_center = int(0.7 * self.last_position[1] + 0.3 * y_center)
                    
                    x = max(x_center - initial_w // 2, 0)
                    y = max(y_center - initial_h // 2, 0)
                    w = initial_w
                    h = initial_h
                    self.track_window = (x, y, w, h)
                    self.last_position = (x_center, y_center)
                    return self.track_window

        return None

def main():
    # Load YOLOv8 model
    model = YOLO('F:\\Desktop\\new data det\\nimra\\nimra.pt')  # Update this to your model path

    # Open webcam
    cap = cv2.VideoCapture(0)  # Use the default webcam

    if not cap.isOpened(): 
        print("Error: Could not open webcam.")
        exit()

    # Initialize tracker variable
    tracker = None
    tracker_initialized = False

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Make predictions
        results = model(frame)
        
        # Extract detections
        detections = results[0].boxes.xyxy.numpy()  # Bounding boxes (x1, y1, x2, y2)
        confidences = results[0].boxes.conf.numpy()  # Confidence scores

        if len(detections) > 0:
            # Determine the best detection
            max_confidence_idx = np.argmax(confidences)  # Index of the highest confidence score
            x1, y1, x2, y2 = detections[max_confidence_idx]
            confidence = confidences[max_confidence_idx]
            bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

            if confidence > 0.5:  # Only initialize if confidence is above a threshold
                if not tracker_initialized:
                    # Initialize the BGT tracker with the detected object
                    tracker = BGTTracker()
                    tracker.initialize(frame, bbox)
                    tracker_initialized = True
                else:
                    # Update the tracker with the current frame
                    track_window = tracker.update(frame)
                    
                    if track_window:
                        x, y, w, h = track_window
                        p1 = (x, y)
                        p2 = (x + w, y + h)
                        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                        cv2.putText(frame, f"Tracking: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    else:
                        cv2.putText(frame, "Lost", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "No Object Detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            if tracker_initialized:
                # Continue tracking if no detection is found
                track_window = tracker.update(frame)
                if track_window:
                    x, y, w, h = track_window
                    p1 = (x, y)
                    p2 = (x + w, y + h)
                    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                    cv2.putText(frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                else:
                    cv2.putText(frame, "Lost", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    tracker_initialized = False
            else:
                cv2.putText(frame, "No Object Detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame with bounding boxes and tracking information
        cv2.imshow('Tracking', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
