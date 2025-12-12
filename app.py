import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import tempfile
import os

st.set_page_config(page_title="Correct Pose Detection", layout="wide")

# Mediapipe setup (CPU-friendly)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return round(angle, 2)

# ================= SESSION STORAGE =================
if "run_cam" not in st.session_state:
    st.session_state.run_cam = False

if "record_frames" not in st.session_state:
    st.session_state.record_frames = []

if "csv_data" not in st.session_state:
    st.session_state.csv_data = []

# keep counters and statuses in session (optional)
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

# Right side windows for data
with right_col:
    reps_box = st.empty()
    stage_box = st.empty()
    posture_box = st.empty()
    angle_box = st.empty()

# ========================= CAMERA INPUT (browser) =========================
# When run_cam is True we show a browser webcam widget and process frames as snapshots.
if st.session_state.run_cam:
    # Show instruction
    left_col.info("Webcam is active. Allow browser camera access if prompted. "
                  "This uses the browser webcam (works on Streamlit Cloud).")

    # camera_input returns an uploaded-like file (snapshot). key ensures widget persistence.
    img_file_buffer = left_col.camera_input("Camera (browser)", key="camera")

    if img_file_buffer is not None:
        # Read image bytes into OpenCV format
        file_bytes = np.frombuffer(img_file_buffer.getvalue(), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # BGR
        if img_bgr is None:
            st.error("Could not read camera frame.")
        else:
            # Convert to RGB for mediapipe processing
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Process frame using MediaPipe (CPU)
            with mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ) as pose:
                results = pose.process(img_rgb)

            # If landmarks found, run the same logic you already had
            if results and results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # --- Left arm joints (used for bicep curl) ---
                # Note: Using same indices/logic as your original code
                try:
                    # LEFT ARM (for bicep curl logic: left side)
                    r_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    r_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    r_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    # RIGHT ARM (using RIGHT_* landmarks as in your original)
                    l_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    l_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    l_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                    # Calculate angles (preserve your original naming)
                    r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
                    l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)

                    # Save angles to session
                    st.session_state.l_angle = l_angle
                    st.session_state.r_angle = r_angle

                    # === Exercise logic (Bicep curl) ===
                    if exercise == "Bicep curl":
                        # Posture classification – same thresholds
                        if l_angle > 170:
                            st.session_state.pose_status = "Bad Pose – Arm hyperextended"
                        elif 120 <= l_angle <= 170:
                            st.session_state.pose_status = "Good Pose – Down position"
                        elif 70 <= l_angle < 120:
                            st.session_state.pose_status = "Good Pose – Moving"
                        elif 30 <= l_angle < 70:
                            st.session_state.pose_status = "Good Pose – Up position"
                        elif l_angle < 20:
                            st.session_state.pose_status = "Bad Pose – Arm over-contracted"
                        else:
                            st.session_state.pose_status = "Good Pose"

                        # Rep counting – same as original
                        if l_angle >= 120:
                            st.session_state.stage = "down"
                        if l_angle <= 40 and st.session_state.stage == "down":
                            st.session_state.stage = "up"
                            st.session_state.counter += 1

                    # === Exercise logic (Lateral raise) ===
                    if exercise == "Lateral raise":
                        r_shoulder2 = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                       landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        r_elbow2 = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        r_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                        angle_lr = calculate_angle(r_hip, r_shoulder2, r_elbow2)
                        st.session_state.l_angle = angle_lr

                        if angle_lr > 160:
                            st.session_state.pose_status = "Bad Pose – Arm too extended"
                        elif angle_lr < 20:
                            st.session_state.pose_status = "Bad Pose – Arm too contracted"
                        else:
                            st.session_state.pose_status = "Good Pose"

                        if angle_lr < 30:
                            st.session_state.stage = "down"
                        if angle_lr > 80 and st.session_state.stage == "down":
                            st.session_state.stage = "up"
                            st.session_state.counter += 1

                    # Draw landmarks on the RGB frame and convert back to BGR for storage
                    annotated_rgb = img_rgb.copy()
                    mp_drawing.draw_landmarks(annotated_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

                    # ================= SAVE FRAME & CSV (same format as before) =================
                    st.session_state.record_frames.append(annotated_bgr.copy())

                    st.session_state.csv_data.append([
                        exercise,
                        st.session_state.counter,
                        st.session_state.stage,
                        st.session_state.pose_status,
                        st.session_state.l_angle,
                        st.session_state.r_angle
                    ])

                    # Show annotated image in the left frame window
                    frame_window.image(annotated_rgb, channels="RGB")

                except Exception as e:
                    # Keep behaviour silent like original; log to stderr optionally
                    # st.error(f"Processing error: {e}")  # uncomment for debugging
                    pass
            else:
                # No landmarks detected — show the raw frame
                frame_window.image(img_rgb, channels="RGB")
    else:
        # No camera access yet; show placeholder
        frame_window.image(np.zeros((480, 640, 3), dtype=np.uint8), channels="BGR")

else:
    # When camera is not running, show placeholder or the last saved frame
    if len(st.session_state.record_frames) > 0:
        last = st.session_state.record_frames[-1]
        last_rgb = cv2.cvtColor(last, cv2.COLOR_BGR2RGB)
        frame_window.image(last_rgb, channels="RGB")
    else:
        frame_window.image(np.zeros((480, 640, 3), dtype=np.uint8), channels="BGR")

# ========================= Right side metrics =========================
reps_box.markdown(f"### Reps: **{st.session_state.counter}**")
stage_box.markdown(f"### Stage: **{st.session_state.stage}**")
posture_box.markdown(f"### Posture: **{st.session_state.pose_status}**")
angle_box.markdown(f"### Angle: **{st.session_state.l_angle}°**")

# ======================= VIDEO DOWNLOAD =======================
if upload and len(st.session_state.record_frames) > 0:

    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

    # Use the shape of the first frame
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

    # Clear saved frames after download to match previous behaviour
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
