import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import time

st.set_page_config(page_title="Correct Pose Detection", layout="wide")

# Mediapipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return round(angle, 2)

# ------------------- SESSION STORAGE -------------------
if "csv_rows" not in st.session_state:
    st.session_state.csv_rows = []

if "rep_count" not in st.session_state:
    st.session_state.rep_count = 0

if "stage" not in st.session_state:
    st.session_state.stage = None

if "exercise" not in st.session_state:
    st.session_state.exercise = "Bicep curl"

# ------------------- UI -------------------
st.title("Correct Pose Detection - Real-time WebRTC")

exercise = st.selectbox("Select Exercise", ["Bicep curl", "Lateral raise"])
st.session_state.exercise = exercise

reps_box = st.empty()
stage_box = st.empty()
posture_box = st.empty()
angle_box = st.empty()

# ============================================================
#   VIDEO PROCESSOR CLASS — REAL-TIME MEDIAPIPE POSE DETECTION
# ============================================================
class PoseVideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self.pose.process(img_rgb)

        l_angle = 0
        r_angle = 0
        pose_status = ""

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark

            # -------------------- BICEP CURL --------------------
            if st.session_state.exercise == "Bicep curl":
                try:
                    # Right arm
                    shoulder = [landmarks[11].x, landmarks[11].y]
                    elbow = [landmarks[13].x, landmarks[13].y]
                    wrist = [landmarks[15].x, landmarks[15].y]
                    l_angle = calculate_angle(shoulder, elbow, wrist)

                    if l_angle > 170:
                        pose_status = "Bad Pose – Arm hyperextended"
                    elif 120 <= l_angle <= 170:
                        pose_status = "Good – Down"
                    elif 30 <= l_angle < 120:
                        pose_status = "Good – Moving"
                    elif l_angle < 30:
                        pose_status = "Good – Up"
                    else:
                        pose_status = "Good Pose"

                    # Rep counter
                    if l_angle >= 120:
                        st.session_state.stage = "down"
                    if l_angle <= 40 and st.session_state.stage == "down":
                        st.session_state.stage = "up"
                        st.session_state.rep_count += 1

                except:
                    pass

            # -------------------- LATERAL RAISE --------------------
            if st.session_state.exercise == "Lateral raise":
                try:
                    shoulder = [landmarks[11].x, landmarks[11].y]
                    elbow = [landmarks[13].x, landmarks[13].y]
                    hip = [landmarks[23].x, landmarks[23].y]

                    l_angle = calculate_angle(hip, shoulder, elbow)

                    if l_angle > 160:
                        pose_status = "Bad – Over extended"
                    elif l_angle < 20:
                        pose_status = "Bad – Too contracted"
                    else:
                        pose_status = "Good"

                    if l_angle < 30:
                        st.session_state.stage = "down"
                    if l_angle > 80 and st.session_state.stage == "down":
                        st.session_state.stage = "up"
                        st.session_state.rep_count += 1

                except:
                    pass

        # ------------------ UPDATE UI ------------------
        reps_box.markdown(f"### Reps: **{st.session_state.rep_count}**")
        stage_box.markdown(f"### Stage: **{st.session_state.stage}**")
        posture_box.markdown(f"### Posture: **{pose_status}**")
        angle_box.markdown(f"### Angle: **{l_angle}°**")

        # ------------------ SAVE CSV DATA ------------------
        st.session_state.csv_rows.append([
            st.session_state.exercise,
            st.session_state.rep_count,
            st.session_state.stage,
            pose_status,
            l_angle,
            r_angle,
            time.time()
        ])

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ============================================================
#                  START WEBRTC STREAM
# ============================================================
webrtc_streamer(
    key="pose-detection",
    mode=WebRtcMode.LIVE,
    video_processor_factory=PoseVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

# ============================================================
#                  DOWNLOAD CSV BUTTON
# ============================================================
if st.button("Download Details CSV"):
    df = pd.DataFrame(st.session_state.csv_rows, columns=[
        "Exercise", "Reps", "Stage", "Posture", "Left Angle", "Right Angle", "Timestamp"
    ])
    csv = df.to_csv(index=False)
    st.download_button("Download CSV", csv, "pose_details.csv", "text/csv")
