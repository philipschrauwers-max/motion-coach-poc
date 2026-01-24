# Motion Coach POC (Savate)

A proof-of-concept motion tracking and coaching app for **Savate**, built with:
- Python
- MediaPipe Pose
- OpenCV

## Features
- Real-time pose tracking via webcam
- Jab / Cross / Fouetté detection
- Calibration-based scoring
- Session export (JSON / CSV)
- Basic performance plotting

## Requirements
- Python 3.10 (recommended)
- Webcam

## Run
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py