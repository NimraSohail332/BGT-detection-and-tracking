import cv2
import numpy as np
from ultralytics import YOLO

class AccurateBackgroundGradientTracker:
    def __init__(self):
        self.roi_hist = None
        self.track_window = None

    def initialize(self, frame, bbox):
        """ 
        Initialize the tracker with the first frame and the bounding box.
        """
        x, y, w, h = bbox
        self.track_window = (x, y, w, h)
        
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            raise ValueError("ROI is empty. Ensure the ROI is within the frame boundaries.")
        
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Calculate and normalize the histogram
        self.roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [16, 16], [0, 256, 0, 256])
        cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)

    def update(self, frame):
        """
        Update the tracker with the current frame and return the new position of the bounding box.
        """
        if self.roi_hist is None or self.track_window is None:
            return None

        x, y, w, h = self.track_window
        
        # Calculate back projection
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        back_proj = cv2.calcBackProject([hsv_frame], [0, 1], self.roi_hist, [0, 256, 0, 256], 1)
        
        # Apply BGT algorithm to find the new position of the object
        mask = cv2.inRange(hsv_frame, (0, 60, 32), (180, 255, 255))
        back_proj &= mask
        
        ret, new_track_window = cv2.meanShift(back_proj, self.track_window, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1))

        if ret:
            self.track_window = new_track_window
            return new_track_window
        else:
            return None

def main():
    # Load YOLOv8 model
    model = YOLO('F:\\Desktop\\new data det\\nimra\\nimra.pt')  # Update this to your model path

    # Open webcam
    cap = cv2.VideoCapture(0)

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
            print("Error: Could not read frame.")
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

            if not tracker_initialized:
                # Initialize the BGT tracker with the detected object
                tracker = AccurateBackgroundGradientTracker()
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

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
