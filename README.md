# Gesture-Based Computer Automation

Control your computer using hand gestures through your webcam! This project uses **OpenCV**, **MediaPipe**, and **Python** to recognize real-time hand gestures and perform system actions like mouse clicks, media control, and more.

## 🚀 Features

* Real-time gesture detection using MediaPipe
* Map gestures to system actions (volume, mouse, slides, etc.)
* Easy to train custom gestures
* Works cross-platform (Windows, macOS, Linux)

## 🧠 Tech Stack

* Python, OpenCV, MediaPipe
* TensorFlow / PyTorch (for model training)
* pynput / pyautogui (for automation)

## 🛠️ Installation

```bash
git clone https://github.com/<your-username>/gesture-based-computer-automation.git
cd gesture-based-computer-automation
pip install -r requirements.txt
```

## ▶️ Usage

```bash
python src/infer.py
```

Perform gestures like **thumbs up**, **fist**, or **open palm** to trigger actions.

## 📦 Project Structure

```
src/
├─ capture.py      # Camera input & detection
├─ model.py        # Gesture classification
├─ infer.py        # Run automation
```


