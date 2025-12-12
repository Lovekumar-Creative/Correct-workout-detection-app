import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import tempfile
from datetime import datetime

st.set_page_config(page_title="Correct Pose Detection", layout="wide")

# Mediapipe setup (CPU only)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle
    return round(angle, 2)

# ================= SESSION STATE =================
if "counter" not in st.session_state:
    st.session_state.counter = 0
if "stage" not in st.session_state:
    st.session_state.stage = None
if "pose_status" not in st.session_state:
    st.session_state.pose_status = ""
if "l_angle" not in st.session_state:
    st.session_state.l_angle = 0
if "r_angle" not in st.session_state:
    st.session_state.r_angle = 0
if "csv_data" not in st.session_state:
    st.session_state.csv_data = []

st.title("Correct Pose Detection App (Cloud Compatible)")

exercise = st.selectbox(
    "Select Your Exercise",
    ["Bicep curl", "Lateral raise"]
)

# ================= CAMERA INPUT =================
st.subheader("Webcam Input (Cloud Compatible)")
img_file_buffer = st.camera_input("Enable webcam to begin exercise")

if img_file_buffer is not None:
    frame = cv2.imdecode(np.frombuffer(img_file_buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ================= PROCESS USING MEDIAPIPE CPU =================
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as pose:

        results = pose.process(frame)

        if results.pose_landmarks:

            landmarks = results.pose_landmarks.landmark

            # Common joints
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # ================= BICEP CURL =================
            if exercise == "Bicep curl":
                angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                st.session_state.l_angle = angle

                if angle >= 120:
                    st.session_state.stage = "down"
                if angle <= 40 and st.session_state.stage == "down":
                    st.session_state.stage = "up"
                    st.session_state.counter += 1

                if angle > 170:
                    st.session_state.pose_status = "Bad Pose – Arm hyperextended"
                elif 120 <= angle <= 170:
                    st.session_state.pose_status = "Good Down Position"
                elif 70 <= angle < 120:
                    st.session_state.pose_status = "Good Motion"
                elif 20 <= angle < 70:
                    st.session_state.pose_status = "Up Position"
                else:
                    st.session_state.pose_status = "Over Contracted"

            # ================= LATERAL RAISE =================
            if exercise == "Lateral raise":
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                angle = calculate_angle(hip, left_shoulder, left_elbow)
                st.session_state.l_angle = angle

                if angle < 30:
                    st.session_state.stage = "down"
                if angle > 80 and st.session_state.stage == "down":
                    st.session_state.stage = "up"
                    st.session_state.counter += 1

                if angle > 160:
                    st.session_state.pose_status = "Bad – Arm too extended"
                elif angle < 20:
                    st.session_state.pose_status = "Bad – Too low"
                else:
                    st.session_state.pose_status = "Good Pose"

            # Draw Pose
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Save CSV row
            st.session_state.csv_data.append([
                datetime.now(),
                exercise,
                st.session_state.counter,
                st.session_state.stage,
                st.session_state.pose_status,
                st.session_state.l_angle,
                st.session_state.r_angle
            ])

# ================= SHOW OUTPUT =================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Pose Visualization")
    if img_file_buffer is not None:
        st.image(frame, channels="RGB")

with col2:
    st.subheader("Exercise Metrics")
    st.write(f"### Reps: **{st.session_state.counter}**")
    st.write(f"### Stage: **{st.session_state.stage}**")
    st.write(f"### Status: **{st.session_state.pose_status}**")
    st.write(f"### Angle: **{st.session_state.l_angle}°**")

# =================== DOWNLOAD CSV ===================
if st.button("Download Exercise Data") and len(st.session_state.csv_data) > 0:
    df = pd.DataFrame(st.session_state.csv_data, columns=[
        "Time",
        "Exercise",
        "Reps",
        "Stage",
        "Pose Status",
        "Left Angle",
        "Right Angle"
    ])

    temp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(temp_csv.name, index=False)

    with open(temp_csv.name, "rb") as f:
        st.download_button(
            label="Download CSV File",
            data=f,
            file_name="exercise_data.csv",
            mime="text/csv"
        )
