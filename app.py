import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import tempfile
import os

st.set_page_config(page_title="Correct Pose Detection", layout="wide")

# ================== REAL CLOUD DETECTION (RELIABLE) ==================
# Streamlit Cloud does NOT have /dev/video0 → local machines DO.
has_camera = os.path.exists("/dev/video0")
# =====================================================================

# Mediapipe setup
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

# ================= SESSION STORAGE =================
if "run_cam" not in st.session_state:
    st.session_state.run_cam = False

if "record_frames" not in st.session_state:
    st.session_state.record_frames = []

if "csv_data" not in st.session_state:
    st.session_state.csv_data = []

# UI Title
st.title("Correct Pose Detection App")

exercise = st.selectbox(
    "Select Your Exercise",
    ["Walking", "Bicep curl", "Lateral raise"]
)

# ========================= BUTTONS =========================
col1, col2, col3, col4 = st.columns([1,1,1,1])

with col1:
    start = st.button("▶ Start Camera")
with col2:
    stop = st.button("⏹ Stop Camera")
with col3:
    upload = st.button("Output Video")
with col4:
    details = st.button("Download Details")

if start:
    st.session_state.run_cam = True
if stop:
    st.session_state.run_cam = False

# ========================= HORIZONTAL LAYOUT =========================
left_col, right_col = st.columns([2, 1])  # Camera left, settings right

frame_window = left_col.empty()          # Camera feed on left side

# Right side info panels
with right_col:
    reps_box = st.empty()
    stage_box = st.empty()
    posture_box = st.empty()
    angle_box = st.empty()

# ========================= CAMERA LOOP =========================
# SAFETY GUARD: BLOCK camera on Streamlit Cloud
if st.session_state.run_cam and not has_camera:
    st.warning("⚠️ Webcam is not supported on Streamlit Cloud.\n"
               "Please run this app on your **local computer** to use the camera.")
    st.stop()

if st.session_state.run_cam and has_camera:

    cap = cv2.VideoCapture(0)

    counter = 0
    stage = None
    pose_status = ""
    l_angle = 0
    r_angle = 0

    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    while st.session_state.run_cam:

        ret, img = cap.read()
        if not ret:
            st.error("Failed to access camera.")
            break

        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        results = pose.process(img)
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # ================= EXERCISE LOGIC =================
        if exercise == "Bicep curl":
            try:
                landmarks = results.pose_landmarks.landmark

                # LEFT ARM
                r_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                r_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                r_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)

                # RIGHT ARM
                l_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                l_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                l_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)

                # REP LOGIC
                if l_angle >= 120:
                    stage = "down"
                if l_angle <= 40 and stage == "down":
                    stage = "up"
                    counter += 1

                # POSTURE LOGIC
                if l_angle > 170:
                    pose_status = "Bad Pose – Arm hyperextended"
                elif 120 <= l_angle <= 170:
                    pose_status = "Good Pose – Down position"
                elif 70 <= l_angle < 120:
                    pose_status = "Good Pose – Moving"
                elif 30 <= l_angle < 70:
                    pose_status = "Good Pose – Up position"
                elif l_angle < 20:
                    pose_status = "Bad Pose – Arm over-contracted"

            except:
                pass

            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # ============== LATERAL RAISE LOGIC =================
        if exercise == "Lateral raise":
            try:
                landmarks = results.pose_landmarks.landmark

                r_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                r_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                r_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                r_angle = calculate_angle(r_hip, r_shoulder, r_elbow)

                l_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                l_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                l_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                l_angle = calculate_angle(l_hip, l_shoulder, l_elbow)

                if l_angle < 30:
                    stage = "down"
                if l_angle > 80 and stage == "down":
                    stage = "up"
                    counter += 1

            except:
                pass

            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # ================= SAVE FRAME =================
        st.session_state.record_frames.append(img.copy())

        st.session_state.csv_data.append([
            exercise,
            counter,
            stage,
            pose_status,
            l_angle,
            r_angle
        ])

        # ================= UI UPDATE =================
        reps_box.markdown(f"### Reps: **{counter}**")
        stage_box.markdown(f"### Stage: **{stage}**")
        posture_box.markdown(f"### Posture: **{pose_status}**")
        angle_box.markdown(f"### Angle: **{l_angle}°**")

        frame_window.image(img, channels="BGR")

        if not st.session_state.run_cam:
            break

    cap.release()
    frame_window.empty()

# ======================= VIDEO DOWNLOAD =======================
if upload and len(st.session_state.record_frames) > 0:

    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

    height, width, _ = st.session_state.record_frames[0].shape
    writer = cv2.VideoWriter(
        temp_video.name,
        cv2.VideoWriter_fourcc(*"mp4v"),
        20,
        (width, height)
    )

    for frame in st.session_state.record_frames:
        writer.write(frame)

    writer.release()

    with open(temp_video.name, "rb") as video_file:
        st.download_button(
            label="Output Video",
            data=video_file,
            file_name="output_video.mp4",
            mime="video/mp4"
        )

    st.session_state.record_frames = []

# ======================= CSV DOWNLOAD =======================
if details and len(st.session_state.csv_data) > 0:

    df = pd.DataFrame(st.session_state.csv_data, columns=[
        "Exercise",
        "Reps",
        "Stage",
        "Posture Status",
        "Left Angle",
        "Right Angle"
    ])

    temp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(temp_csv.name, index=False)

    with open(temp_csv.name, "rb") as f:
        st.download_button(
            label="Download Details",
            data=f,
            file_name="pose_details.csv",
            mime="text/csv"
        )

    st.session_state.csv_data = []
