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

Here’s the section you can copy and paste directly into your README file, formatted with headings and bullet points:

---

## Training

### Prepare Your Dataset

- Organize images and label files into appropriate directories (e.g., `images/` and `labels/`).

### Configure YOLOv8

- Set up a configuration file for training, specifying paths to your data, classes, and hyperparameters.

### Train the Model

- Use the YOLOv8 training script to train the model on your dataset. For example:

  ```bash
  yolo train --data your_data.yaml --cfg your_model.cfg --weights '' --batch-size 16 --epochs 50
  ```

- Adjust parameters based on your dataset and needs.

## Evaluation Metrics

### Confusion Matrix

The confusion matrix provides insight into the classification performance of the model. It shows:

- **True Positives (TP):** Correctly identified positive instances.
- **True Negatives (TN):** Correctly identified negative instances.
- **False Positives (FP):** Incorrectly identified positive instances.
- **False Negatives (FN):** Incorrectly identified negative instances.
  ![Confusion Matrix](https://github.com/NimraSohail332/BGT-detection-and-tracking/blob/main/train17/F1_curve.png)

A balanced confusion matrix indicates good performance, while imbalances highlight areas needing improvement.

### F1 Score Curve

The F1 score curve shows the trade-off between precision and recall across different threshold values. The F1 score is the harmonic mean of precision and recall:

- **Precision:** The ratio of true positives to the total predicted positives.
- **Recall:** The ratio of true positives to the total actual positives.
- **F1 Score:** Balances precision and recall into a single metric.

A higher F1 score indicates better overall performance, especially important in cases with class imbalances.

---


   

## Results

- **Detection Performance:** Evaluated based on precision, recall, and accuracy of YOLOv8 on the validation dataset.
- **Tracking Performance:** Assessed on the ability of BGT to maintain accurate tracking across frames.


