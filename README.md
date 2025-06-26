# Poisonous Plant and Soil Type Detection

This project is a computer vision system for detecting poisonous plants and classifying soil types in real-time using screenshots or live camera feed. It integrates object detection (YOLOv8) with two FastAI-based classifiers for accurate identification and provides optional GPS geolocation for field threat tracking.

## Features

* Real-time object detection using YOLOv8
* Classification of poisonous plants (e.g., hemlock, foxglove, oleander)
* Classification of soil types (e.g., sandy, clay, loamy)
* GPS data retrieval via HTTP
* Visual output with confidence scores and bounding boxes

## Technologies Used

* Python
* OpenCV
* Ultralytics YOLOv8
* FastAI
* Torch & TorchVision
* PIL (Pillow)
* Requests

## Installation & Setup

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/plant-soil-detector.git
cd plant-soil-detector
```

2. **(Optional) Create virtual environment:**

```bash
python -m venv .venv
source .venv/bin/activate    # or .venv\Scripts\activate on Windows
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Download model weights and place them:**

The following model files are **not included** in the repository due to GitHub file size limits:

- `models/yolov8n.pt`
- `models/stage-1.pth`
- `models/soil_model.pkl`

Please download them manually from [Google Drive](https://drive.google.com/drive/folders/116NMJBn8ydc00lUNtFmMmlBPgrnlADqG?usp=sharing) and place them into the `models/` folder.

## How It Works

* The system captures frames via webcam or emulator.
* YOLOv8 detects objects and returns bounding boxes.
* Cropped images of detected regions are passed to one of two classifiers:

  * **Plant Classifier:** Determines if the plant is poisonous.
  * **Soil Classifier:** Classifies the soil type.
* If a poisonous plant is found with confidence > 70%, an alert is triggered.
* Optional GPS coordinates are retrieved if available.

## Plant Classes Detected

* Deadly Nightshade
* Hemlock
* Oleander
* Castor Oil Plant
* Foxglove

> Other plants will be marked "Safe".

## Soil Types Detected

* black soil 
* chestnut soil 
* grey soil 
* red soil 
* saline 
* sand
* no soil (if no soil detected)


## Project Structure

```
plant-soil-detector/
├── main.py
├── models/
│   ├── yolov8l.pt
│   ├── stage-1.pth
│   └── soil_model.pkl
├── requirements.txt
├── README.md
├── LICENSE
├── screenshots/
```

## Sample Results

### Soil Type Classification

![Soil Type Classification](https://i.ibb.co/8Dcxj7rX)

### Detection with Confidence Output

![Detection 2](https://i.ibb.co/GQ7cx9Mh)

### Poisonous Plant Detection

![Poisonous Plant Detection 1](https://i.ibb.co/TqpqPxHW)

## Notes

* Model weights are too large for GitHub. Download them separately and place in `models/`.
* BlueStacks or webcam is required to provide input for analysis.
* Classifier training scripts are not included but can be provided on request.

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/Iluxa-sensei/Plant-Soil-Detector/blob/main/MIT%20LICENSE)file for details.

## Contact

For questions, feedback, or collaboration, open an issue or contact via GitHub: [github.com/your-username/plant-soil-detector](https://github.com/your-username/plant-soil-detector/issues)
