<p align="center">
  <img src="https://raw.githubusercontent.com/bkai-ndt-sdh231/.github/main/profile/BKAI_logo.png" width="180">
</p>

<h1 align="center">BKAI – Mask R-CNN + ResNet50 for Concrete Crack Detection & Segmentation</h1>

Official repository for the BKAI deep-learning model for **detecting and segmenting concrete cracks** using **Mask R-CNN + ResNet-50 + FPN**.  
This model is part of the *BKAI Smart Infrastructure Initiative*.

---

## 🔗 GitHub Repository

👉 <https://github.com/bkai-ndt-sdh231/BKAI-Model-Mask-R-CNN>

---

## 📌 Model Overview

- **Architecture**: Mask R-CNN (ResNet-50 + FPN)  
- **Task**: Concrete crack detection & instance segmentation  
- **Framework**: Detectron2 (PyTorch)  
- **Number of classes**: 1 (`crack`)  
- **Dataset size**: 12,000+ images (collected in Vietnam & Japan)  
- **Annotation format**: COCO instance segmentation  

---

## 📊 Evaluation Summary

### Detection (Bounding Box)

| Metric            | Score   |
|-------------------|---------|
| **mAP (0.5:0.95)**| **63.99%** |
| **AP50**          | **82.89%** |
| **AP75**          | **72.00%** |

### Segmentation (Mask)

| Metric            | Score   |
|-------------------|---------|
| **mAP (0.5:0.95)**| **21.53%** |
| **AP50**          | **50.77%** |
| **AP75**          | **17.36%** |
### Classification (Crack / No Crack)

| Metric        | Value  |
|---------------|--------|
| **F1-score**  | **99.13%** |
| **Recall**    |**99.89%**  |
| **Precision** | **98.37%** |
| **Accuracy**  | **98%**    |
---

## 📁 Files Included

- `mask_rcnn_resnet50_v7.pth` – Trained final model weights  
- `metrics_v7.json` – Full COCO evaluation metrics  
- `sample_pairs_crack_20/` – Before/after visualization pairs  
- `BKAI_Results.rar` – Training charts (loss curves, mAP, PR curve, etc.)  

---

## 🛠 Quick Inference Example

```python
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import cv2

# 1. Build config
cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
)

# 2. Update for BKAI crack model
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1          # only "crack"
cfg.MODEL.WEIGHTS = "mask_rcnn_resnet50_v7.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # confidence threshold
cfg.MODEL.DEVICE = "cuda"  # or "cpu"

# 3. Create predictor
predictor = DefaultPredictor(cfg)

# 4. Run inference
img = cv2.imread("test.jpg")
outputs = predictor(img)
print(outputs)  # instances.pred_boxes, instances.scores, instances.pred_masks, ...


@misc{bkai2025-crack-maskrcnn,
  title  = {BKAI – Mask R-CNN + ResNet50 for Concrete Crack Detection & Segmentation},
  author = {Nguyen Dat Thanh},
  year   = {2025},
  url    = {https://github.com/bkai-ndt-sdh231/BKAI-Model-Mask-R-CNN}
}

## 📸 Example Visualization

### Example 1  
**Input**  
<img src="examples/example1_input.png" width="45%">

**Overlay (Mask R-CNN result)**  
<img src="examples/example1_overlay.png" width="45%">

---

### Example 2  
**Input**  
<img src="examples/example2_input.png" width="45%">

**Overlay (Mask R-CNN result)**  
<img src="examples/example2_overlay.png" width="45%">

---

### Example 3  
**Input**  
<img src="examples/example3_input.png" width="45%">

**Overlay (Mask R-CNN result)**  
<img src="examples/example3_overlay.png" width="45%">

---

## 🧪 Training Details

- **Batch size:** 2 (Colab GPU)  
- **Learning rate:** 0.00015  
- **Iterations:** 8,000  

**Loss components:**
- RPN objectness loss  
- RPN localization loss  
- ROI classification loss  
- ROI bounding-box regression  
- Mask binary cross-entropy  

---

## 🏗 Applications

- Infrastructure monitoring  
- Pavement/bridge crack inspection  
- Construction quality control  
- BIM + AI  
- Post-earthquake damage assessment  

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 🔖 Citation

```bibtex
@misc{bkai2025-crack-maskrcnn,
  title  = {BKAI – Mask R-CNN + ResNet50 for Concrete Crack Detection & Segmentation},
  author = {Nguyen Dat Thanh},
  year   = {2025},
  url    = {https://github.com/bkai-ndt-sdh231/BKAI-Model-Mask-R-CNN}
}


