# 🏋️ Correct Workout Pose Detection App  
### Real-Time Exercise Form Analysis using Computer Vision

---

## 📌 Project Overview

The **Correct Workout Pose Detection App** is a real-time computer vision–based fitness application that analyzes human posture during workouts using a webcam.  
It detects **incorrect exercise form**, **counts repetitions automatically**, and provides **instant posture feedback** to help users exercise safely and effectively.

This project was built as an **end-to-end AI application**, covering real-time video processing, pose estimation, UI design, and data export.

---

## 🎯 Motivation

Many people perform workouts at home without professional guidance, leading to:
- Incorrect posture
- Higher injury risk
- Inefficient training

This project solves that problem by using **AI-powered pose detection** to act as a **virtual fitness assistant**, requiring only a webcam—no wearable devices.

---

## 🚀 Features

- 🎥 Real-time webcam-based pose detection  
- 🧍 Exercise-specific posture validation  
- 🔢 Automatic repetition counting  
- ⚠️ Good / Bad posture feedback  
- 📐 Joint angle measurement  
- 📊 Live workout stats (Reps, Stage, Angles)  
- 📁 CSV workout report download  
- 🎞️ Output video export  
- 🖥️ Clean horizontal UI layout using Streamlit  

---

## 🏃 Supported Exercises

| Exercise | Analysis Performed |
|--------|-------------------|
| Bicep Curl | Elbow angle, arm contraction & extension |
| Lateral Raise | Shoulder raise angle |
| Walking | UI-ready (logic extendable) |

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit** – Web UI & interaction
- **OpenCV** – Video capture & processing
- **MediaPipe Pose** – Human pose landmark detection
- **NumPy** – Angle calculations
- **Pandas** – CSV data generation

---

## 🧠 How It Works

### 1️⃣ Pose Detection
- Uses **MediaPipe Pose** to detect 33 body landmarks per frame.
- Extracts joints like shoulder, elbow, wrist, and hip.

### 2️⃣ Angle Calculation
Joint angles are calculated using vector geometry:

```python
angle = arctan2(...) - arctan2(...)
