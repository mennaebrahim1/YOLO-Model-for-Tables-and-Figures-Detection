# YOLO Model for Tables and Figures Detection

This repository contains the code and dataset for training a YOLO model to detect tables, figures, and equations in documents. The dataset was collected through web scraping and manually labeled using LabelMe. Equations are included under the 'figures' class for streamlined detection.

## Project Overview
The goal of this project is to develop a YOLO-based object detection model capable of identifying and classifying tables and figures (including equations) in images of documents. The trained model can be used for tasks such as:
- Document layout analysis
- Automated table and figure extraction
- Research paper segmentation
- OCR enhancement for structured documents

## Dataset
- **Total Instances:** 2243
- **Classes:**
  - **Figures:** 702 (including equations)
  - **Tables:** 1541
- **Labeling Tool:** LabelMe
- **Annotation Format:** YOLO (bounding box coordinates and class labels)

## Requirements
- Python 3.8+
- PyTorch
- Ultralytics YOLO
- torchvision

Install dependencies with:
```bash
pip install ultralytics torch torchvision
```

## Model Training
The model is trained using YOLOv8 with the following configuration:

```python
from ultralytics import YOLO
import torchvision.ops as ops

# Load a pretrained YOLO model
model = YOLO("yolo11n.pt")

# Apply Non-Maximum Suppression (NMS) if needed
boxes = ops.nms(boxes, scores, iou_thres)

# Train the model
results = model.train(
    data=r"C:\Users\me513\Downloads\yolo\data.yaml",  # Path to dataset YAML
    batch=1,  # Adjust based on GPU capability
    epochs=100,
    imgsz=640,  # Image size for training
    device=0  # GPU (use -1 for CPU)
)
```

## YAML Configuration
Ensure your `data.yaml` file is configured properly with paths to your images and annotations.
```yaml
path: C:/Users/me513/Downloads/yolo  # Root dataset path
train: images/train  # Train images folder
val: images/val  # Validation images folder

nc: 2  # Number of classes
names: ['figure', 'table']
```

## Inference
Run inference on new images after training:
```python
results = model.predict("path/to/image.jpg")
results.show()  # Display predictions
```

## Evaluation
Evaluate the trained model on the validation set:
```python
metrics = model.val()
```

## Acknowledgments
- **Ultralytics YOLO** for the YOLOv8 framework
- **LabelMe** for annotation tools

## License
This project is licensed under the MIT License.

