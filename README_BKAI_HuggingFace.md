<p align="center">
  <img src="https://raw.githubusercontent.com/bkai-ndt-sdh231/.github/main/profile/BKAI_logo.png" width="180">
</p>

#BKAI – Mask R-CNN + ResNet50 for Concrete Crack Detection & Segmentation

Official repository for the BKAI deep-learning model used for **detecting and segmenting concrete cracks** using **Mask R‑CNN + ResNet50 + FPN**.  
This model is part of the *BKAI Smart Infrastructure Initiative*.

---

## 🔗 GitHub Repository  
👉 https://github.com/bkai-ndt-sdh231/BKAI-Model-Mask-R-CNN

---

## 📌 Model Overview
- **Architecture**: Mask R‑CNN (ResNet‑50 FPN)  
- **Task**: Crack detection + segmentation  
- **Framework**: Detectron2 (PyTorch)  
- **Classes**: 1 (`crack`)  
- **Dataset**: 12,000+ images (Vietnam & Japan)  
- **Format**: COCO Segmentation  

---

## 📊 Evaluation Summary

### **Detection (Bounding Box)**
| Metric | Score |
|--------|--------|
| **mAP (0.5:0.95)** | **63.99%** |
| **AP50** | **82.89%** |
| **AP75** | **72.00%** |

### **Segmentation (Mask)**
| Metric | Score |
|--------|--------|
| **mAP (0.5:0.95)** | **21.53%** |
| **AP50** | **50.77%** |

### **Classification-level (Crack/No Crack)**
| Metric | Value |
|--------|--------|
| **F1-score** | **0.991** |
| **Accuracy** | **98%** |

---

## 📁 Files Included
- `mask_rcnn_resnet50_v7.pth` → Trained final model  
- `metrics_v7.json` → Full COCO evaluation  
- `sample_pairs_crack_20/` → Before/after visualization  
- `BKAI_Results.rar` → All charts (mAP, PR curve, etc.)  

---

## 🛠 Quick Inference Example

```python
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import cv2

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.WEIGHTS = "mask_rcnn_resnet50_v7.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cuda"   # or "cpu"

predictor = DefaultPredictor(cfg)

img = cv2.imread("test.jpg")
outputs = predictor(img)
print(outputs)
```

---

## 📸 Example Visualization
```
Input Image  →  Crack Mask  →  Final Overlay
```
*(Add images in HuggingFace UI if needed)*

---

## 🧪 Training Details
- **Batch size**: 2 (Colab GPU)
- **Learning rate**: 0.00015  
- **Iterations**: 8,000  
- **Losses**:
  - RPN: objectness + localization  
  - ROI: classification + box regression  
  - Mask head: binary cross-entropy  

---

## 🏗 Applications
✔ Infrastructure monitoring  
✔ Pavement/bridge inspection  
✔ Construction quality control  
✔ BIM + AI workflows  
✔ Post-earthquake assessment  

---

## 📜 License
This project is licensed under the **MIT License**.

---

## 🔖 Citation
If you use this model, please cite:

```
@misc{bkai2025-crack-maskrcnn,
  title  = {BKAI – Mask R-CNN + ResNet50 for Concrete Crack Detection & Segmentation},
  author = {Nguyen Dat Thanh},
  year   = {2025},
  url    = {https://github.com/bkai-ndt-sdh231/BKAI-Model-Mask-R-CNN}
}
```

---

## 👤 Author
**Nguyễn Đạt Thạnh**  
BKAI – Structural Engineering  
📧 Email: bkai.sdh231@gmail.com  
🔗 GitHub: https://github.com/bkai-ndt-sdh231  

---

<p align="center"><b>⭐ If this model is helpful, please give it a star!</b></p>
