# YOLO-RECAP: Reassembly with Channel Attention for Perception

## 1. Overview YOLO-RECAP
YOLO-RECAP is a novel object detection framework built upon YOLOv11, integrating **CARAFE** and **ECA** modules to enhance performance especially for small object detection and complex backgrounds.
![그림1](https://github.com/user-attachments/assets/d9db2e55-4bb7-4fac-a25e-38529d3fc081)


---

## 2. Environment  
```bash
pip install -r requirements.txt

## 3. Model Architecture
YOLO-RECAP integrates CARAFE into the upsampling path and ECA into the neck/head for improved feature reassembly and channel weighting.

<img src="assets/architecture.png" width="700" alt="YOLO-RECAP Architecture">

**Key Components**
- **CARAFE:** Content-aware reassembly for fine-grained upsampling
- **ECA:** Lightweight 1D channel attention for efficient feature recalibration

## 4. Datasets
- **VisDrone2019:** Drone-based small object dataset
- **SKU-110K:** Retail shelf dataset with dense object arrangement
- **Pascal VOC:** General-purpose object dataset
- **DOTA v1:** Aerial dataset for rotated and small objects
