# YOLO Object Detection Web App

Du an nay hien da duoc dua ve ban web + backend YOLO thuan, khong con tich hop Unity.

He thong gom:

- Backend FastAPI xu ly anh va camera.
- YOLO object detection.
- YOLO pose estimation cho nguoi neu model pose kha dung.
- Pseudo-3D tu bbox/keypoint de hien thi chi tiet tren web.
- Tracking ID don gian theo frame.
- Frontend HTML/CSS/JavaScript de upload anh, mo camera va ve overlay.
- Script kiem tra dataset va train custom YOLO model.

## Cau Truc Chinh

```text
BE/
  main.py                 FastAPI endpoints cho image/camera
  detector.py             Load YOLO detection model
  pose_estimator.py       YOLO pose estimation
  pose_3d.py              Pseudo-3D tu bbox/keypoint
  tracker.py              Gan object_id va velocity tuong doi
  camera_service.py       Doc webcam backend bang OpenCV
  train_yolo.py           Train/fine-tune YOLO detection model
  check_dataset.py        Kiem tra cau truc dataset YOLO
  requirements.txt        Dependency Python
  yolov8n.pt              Model detection mac dinh
  yolov8n-pose.pt         Model pose mac dinh

FE/
  index.html              Giao dien web
  app.js                  Goi API, preview media, ve bbox/keypoints
  style.css               Style giao dien

DataSet/
  data.yaml               Cau hinh dataset YOLO
  README.md               Huong dan format dataset
```

## Cai Moi Truong

```powershell
cd D:\Yolo
.\venv\Scripts\activate
pip install -r BE\requirements.txt
```

## Chay Backend

```powershell
cd D:\Yolo\BE
uvicorn main:app --reload
```

Kiem tra backend:

```text
http://127.0.0.1:8000/
http://127.0.0.1:8000/model/status
```

## Chay Frontend

Mo file:

```text
D:\Yolo\FE\index.html
```

Neu trinh duyet dang cache file JS cu, bam:

```text
Ctrl + F5
```

## Luong Xu Ly

```text
FE/index.html
  -> FE/app.js gui file/camera frame den backend
  -> BE/main.py nhan request
  -> detector.py detect object
  -> pose_estimator.py detect pose neu co
  -> pose_3d.py them pseudo-3D
  -> tracker.py gan object_id
  -> tra JSON ve frontend
  -> FE/app.js ve bbox/keypoint len canvas
```

## API Chinh

| Method | Endpoint | Vai tro |
| --- | --- | --- |
| `GET` | `/` | Kiem tra backend dang chay. |
| `GET` | `/model/status` | Tra trang thai detection model va pose model. |
| `POST` | `/detect/image` | Upload anh va tra detection JSON. |
| `GET` | `/camera/frame` | Doc 1 frame tu webcam backend va detect. |

## File Sinh Ra Khi Chay

| Duong dan | Y nghia |
| --- | --- |
| `BE/__pycache__/` | Cache Python, co the xoa. |
| `venv/` | Moi truong ao Python, khong sua thu cong. |
| `runs/` | Output train YOLO neu co. |

## Train Custom YOLO

Kiem tra dataset:

```powershell
cd D:\Yolo
python BE\check_dataset.py --data DataSet\data.yaml
```

Train:

```powershell
cd D:\Yolo
python BE\train_yolo.py --data DataSet\data.yaml --model BE\yolov8n.pt --epochs 50 --imgsz 640
```

Model tot nhat sau train thuong nam tai:

```text
runs/detect/train/weights/best.pt
```

Backend se uu tien model custom hop le o vi tri do truoc khi fallback ve `BE/yolov8n.pt`.

## Ghi Chu

- Du an khong con phu thuoc Unity va khong con gui UDP.
- `position_3d` la pseudo-3D, khong phai toa do 3D metric that.
- Pose model mac dinh chu yeu danh cho human pose.
- Dataset hien can bo sung anh/label that neu muon train custom.
