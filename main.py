import cv2
import os
import torch
import requests
import numpy as np
from ultralytics import YOLO
from fastai.vision.all import *
from torchvision import transforms
from PIL import Image

yolo_model_path = "../PythonProject2/.venv/yolov8l.pt"
plant_classifier_path = "D:/classify/models/stage-1.pth"
soil_model_path = "D:/greenshield/soil_dataset/soil_model.pkl"


poisonous_plants = ["deadly nightshade", "hemlock", "oleander", "castor oil plant", "foxglove"]


yolo_model = YOLO(yolo_model_path)


plant_checkpoint = torch.load(plant_classifier_path, map_location="cpu")
dls = ImageDataLoaders.from_folder("D:/archive/poisonous_plants_dataset", valid_pct=0.2, seed=42, item_tfms=Resize(299))
learn = vision_learner(dls, resnet50, n_out=8)
learn.model.load_state_dict(plant_checkpoint["model"], strict=False)
learn.model.eval()

soil_learn = load_learner(soil_model_path)

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("⚠️ Poisonous plants:", poisonous_plants)


cap = cv2.VideoCapture(0)

def get_gps_coordinates():
    try:
        response = requests.get("http://192.168.4.1/gps", timeout=2)
        data = response.json()
        return data["lat"], data["lng"]
    except Exception as e:
        print(f"Error receiving GPS: {e}")
        return None, None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to receive frame")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = yolo_model(frame_rgb, imgsz=320, conf=0.3, iou=0.4)

    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            class_name = yolo_model.names[int(cls)].lower()
            x1, y1, x2, y2 = map(int, box)
            crop_img = frame_rgb[y1:y2, x1:x2]
            pil_image = Image.fromarray(crop_img)

            if "plant" in class_name:
                with torch.no_grad():
                    pred_class, pred_idx, probs = learn.predict(pil_image)
                    plant_name = str(pred_class).replace("_", " ").lower()
                    confidence = probs[pred_idx].item()

                if plant_name in poisonous_plants and confidence >= 0.7:
                    safety_status = "⚠️ UNSAFE!"
                    lat, lng = get_gps_coordinates()
                    if lat and lng:
                        print(f"📍Threat coordinates: LAT {lat}, LNG {lng}")
                else:
                    safety_status = "✅ Safe"

                print(f"🌿 Plant discovered: {plant_name}")
                print(f"📊 Model Confidence: {confidence:.2%}")
                print(f"⚠️ {safety_status}")
                print("-" * 30)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{plant_name} - {safety_status} ({confidence:.2f})",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            elif "soil" in class_name:
                pred_class, pred_idx, probs = soil_learn.predict(pil_image)
                soil_type = str(pred_class).replace("_", " ").lower()
                soil_confidence = probs[pred_idx].item()

                print(f"🧱 Soil type: {soil_type}")
                print(f"📊 Soil model confidence: {soil_confidence:.2%}")
                print("-" * 30)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, f"{soil_type} ({soil_confidence:.2f})",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    cv2.imshow("YOLOv8 Detection (Plants & Soil)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
