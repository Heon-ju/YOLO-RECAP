# YOLO-RECAP: Reassembly with Channel Attention for Perception

## 1. Overview YOLO-RECAP
YOLO-RECAP is a novel object detection framework built upon **YOLOv11**, integrating **CARAFE** and **ECA** modules to enhance performance, especially for small object detection and complex backgrounds.

![그림1](https://github.com/user-attachments/assets/d9db2e55-4bb7-4fac-a25e-38529d3fc081)

YOLO-RECAP combines **Content-Aware Reassembly of FEatures** and **efficient channel attention** to achieve a balance between detection accuracy and computational efficiency.  
- **CARAFE (Content-Aware ReAssembly of FEatures):** Generates position-specific kernels for fine-grained upsampling and detailed feature reconstruction.  
- **ECA (Efficient Channel Attention):** Uses adaptive 1D convolution to emphasize informative channels while suppressing redundant ones.  

This architecture enhances small-object detection capability while preserving real-time inference performance.

---

## 2. Environment
```bash
pip install -r requirements.txt
```

## 3. Model Training Example
🚀 python(train_CR.py)
```
python train_CR.py --model "ultralytics/cfg/models/11/yolo11n-CR.yaml" --dataset_config "ultralytics-main/ultralytics/cfg/datasets/VisDrone.yaml" --epochs 100 --imgsz 640 --device 0 --output_dir "E:/YOLOv11_train" --run_name "YOLO11CR_VisDrone"
```   
<br></br><br></br><br></br>
🚀 Notebook(.ipynb)
```
from ultralytics import YOLO
from multiprocessing import freeze_support

model_path   = "C:/User/Desktop/YOLO-RECAP/ultralytics/cfg/models/11/yolo11n-CR.yaml"
data_path    = "C:/User/Desktop/YOLO-RECAP/ultralytics/cfg/datasets/VisDrone.yaml"
epochs       = 100
imgsz        = 640
device       = 0
project_path = "E:/YOLOv11_train"
run_name     = "YOLO11CR_VisDrone"

def main():
    model = YOLO(model_path)
    model.train(data=data_path, epochs=epochs, imgsz=imgsz, device=device, project=project_path, name=run_name)

if __name__ == "__main__":
    freeze_support()  
    main()
```
🌸 🍀 🌷 🧸 🪄 🦾 💪 🛠️
