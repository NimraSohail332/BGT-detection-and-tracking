# BGT Detection and Tracking

**Overview**

The project combines the strengths of YOLOv8's powerful object detection capabilities with a custom Background Gradient Tracker (BGT) to track detected objects over time. This hybrid approach allows for efficient and accurate object tracking even in dynamic environments.

## Description

The system operates in two main stages:

1. **Object Detection:** YOLOv8 is used to detect objects in the video frames. The detected objects are provided with bounding boxes and confidence scores.

2. **Object Tracking:** Once an object is detected, the BGT tracker is initialized with the object's bounding box. The tracker then continuously updates the object's position in subsequent frames.

## BGTTracker Class

The `BGTTracker` class is responsible for tracking the object based on background subtraction and morphological operations to reduce noise. It maintains the object's initial size and smooths the position to reduce jitter.

## Code Flow

1. Load the YOLOv8 model.
2. Capture frames from the webcam.
3. Perform object detection using YOLOv8.
4. If a high-confidence detection is found, initialize the BGT tracker.
5. Update the tracker's position in subsequent frames.
6. Display the tracking results in real-time.

## Dataset Description

- **Training Dataset:** Contains annotated images used to train YOLOv8. Labels are created using LabelImg.
- **Validation Dataset:** Used to validate the model's performance and adjust hyperparameters.

## LabelImg

[LabelImg](https://github.com/tzutalin/labelImg) is an open-source graphical image annotation tool.

### Process

1. Open the image using LabelImg.
2. Draw bounding boxes around the objects of interest.
3. Save the annotations in YOLO format (TXT files).

## Training YOLOv8

1. Configure YOLOv8 for your dataset.
2. Run the training process and save the trained model.

## Results

- **Detection Performance:** Evaluated based on precision, recall, and accuracy of YOLOv8 on the validation dataset.
- **Tracking Performance:** Assessed on the ability of BGT to maintain accurate tracking across frames.


