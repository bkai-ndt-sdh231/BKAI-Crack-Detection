<p align="center">
  <img src="https://raw.githubusercontent.com/bkai-ndt-sdh231/.github/main/profile/BKAI_logo.png" width="150">
</p>

<h1 align="center">BKAI – Mask R-CNN + ResNet50 for Concrete Crack Detection & Segmentation</h1>

<p align="center">
  <b>Official Deep Learning Model for Automated Crack Detection in Concrete Infrastructure</b><br>
  Mask R-CNN + ResNet50 + FPN | Detectron2 | COCO Segmentation
</p>

---

## 📌 Overview

This repository contains the complete implementation and trained model for **automatic detection and segmentation of concrete cracks** using:

- **Mask R-CNN**
- **ResNet-50 Backbone**
- **Feature Pyramid Network (FPN)**
- **Detectron2 (Facebook AI Research)**

This model is used in the research project:

> “Application of Deep Convolutional Neural Networks (CNN) for Detecting and Monitoring Concrete Cracks in Infrastructure.”

---

## 📁 Repository Structure


---

## 🧠 Model Architecture

- **Base Model:** Mask R-CNN  
- **Backbone:** ResNet-50  
- **Feature Extractor:** FPN  
- **Framework:** Detectron2  
- **Task:** Instance segmentation (crack mask) + object detection  
- **Number of classes:** 1 (`crack`)  
- **Dataset:** 12,000+ labeled concrete images (Vietnam & Japan)  
- **Annotation format:** COCO Instance Segmentation  

---

## 📊 Evaluation Results (COCO mAP)

### **Bounding Box (Detection)**

| Metric | Score |
|--------|--------|
| **AP (0.5:0.95)** | **63.99%** |
| **AP50** | **82.89%** |
| **AP75** | **72.04%** |
| APS | 3.96% |
| APM | 11.47% |
| APL | 70.67% |

---

### **Segmentation (Mask)**

| Metric | Score |
|--------|--------|
| **AP (0.5:0.95)** | **21.53%** |
| **AP50** | **50.78%** |
| **AP75** | **17.36%** |
| APS | 0.15% |
| APM | 0.72% |
| APL | 30.04% |

---

### **Crack / No Crack Classification Metrics**

| Metric | Value |
|--------|--------|
| **Accuracy** | **98%** |
| **Precision** | 0.99 |
| **Recall** | 1.00 |
| **F1-score** | **0.991** |

**Confusion Matrix**

|                | Pred No Crack | Pred Crack |
|----------------|---------------|------------|
| **GT No Crack** | 77 | 15 |
| **GT Crack** | 1 | 907 |

📌 *High recall (0.998) indicates excellent ability to detect cracks.*

---

## 📸 Example Visualization

### **Example 1**
**Input:**  
<img src="examples/example1_input.png" width="45%">

**Mask R-CNN Result:**  
<img src="examples/example1_overlay.png" width="45%">

---

### **Example 2**
**Input:**  
<img src="examples/example2_input.png" width="45%">

**Mask R-CNN Result:**  
<img src="examples/example2_overlay.png" width="45%">

---

### **Example 3**
**Input:**  
<img src="examples/example3_input.png" width="45%">

**Mask R-CNN Result:**  
<img src="examples/example3_overlay.png" width="45%">

---

## 🛠 Quick Inference Example

```python
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import cv2

cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
)

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.WEIGHTS = "mask_rcnn_resnet50_v7.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cuda"   # or "cpu"

predictor = DefaultPredictor(cfg)
img = cv2.imread("test.jpg")
outputs = predictor(img)

print(outputs)
