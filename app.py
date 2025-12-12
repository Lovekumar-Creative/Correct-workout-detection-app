import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import tempfile
import os
import time
import html

st.set_page_config(page_title="Correct Pose Detection (Realtime)", layout="wide")

# Mediapipe setup (CPU)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return round(angle, 2)

# ---------------- Session state ----------------
if "run_cam" not in st.session_state:
    st.session_state.run_cam = False
if "record_frames" not in st.session_state:
    st.session_state.record_frames = []
if "csv_data" not in st.session_state:
    st.session_state.csv_data = []
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

# ---------------- UI ----------------
st.title("Correct Pose Detection App (Near-Realtime)")

exercise = st.selectbox("Select Your Exercise", ["Walking", "Bicep curl", "Lateral raise"])

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

left_col, right_col = st.columns([2, 1])
frame_window = left_col.empty()

with right_col:
    reps_box = st.empty()
    stage_box = st.empty()
    posture_box = st.empty()
    angle_box = st.empty()

# ---------------- Realtime refresh control ----------------
# Interval in milliseconds (200 ms for Option 1)
REFRESH_MS = 200

# If camera running, show info and camera widget
if st.session_state.run_cam:
    left_col.info("Webcam active — allow browser camera. App auto-refreshes for near-real-time (~200 ms).")

    # camera_input gives a snapshot each time the page loads
    img_buf = left_col.camera_input("Camera (browser)", key="camera_rt")

    if img_buf is not None:
        # Read frame from buffer (BGR)
        file_bytes = np.frombuffer(img_buf.getvalue(), dtype=np.uint8)
        frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            st.error("Could not read camera frame.")
        else:
            # Convert to RGB for mediapipe
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe (CPU)
            with mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ) as pose:
                results = pose.process(frame_rgb)

            # If landmarks detected, apply your exact logic
            if results and results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                try:
                    # LEFT arm landmarks (used for bicep curl original logic)
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    # RIGHT arm landmarks (preserving your naming)
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                    # Angles like original
                    l_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                    r_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

                    st.session_state.l_angle = l_angle
                    st.session_state.r_angle = r_angle

                    # Bicep curl logic (unchanged)
                    if exercise == "Bicep curl":
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

                        if l_angle >= 120:
                            st.session_state.stage = "down"
                        if l_angle <= 40 and st.session_state.stage == "down":
                            st.session_state.stage = "up"
                            st.session_state.counter += 1

                    # Lateral raise logic (unchanged)
                    if exercise == "Lateral raise":
                        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                        angle_lr = calculate_angle(hip, left_shoulder, left_elbow)
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

                    # Draw landmarks on annotated frame
                    annotated = frame_rgb.copy()
                    mp_drawing.draw_landmarks(annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

                    # Save for video & CSV (same format)
                    st.session_state.record_frames.append(annotated_bgr.copy())
                    st.session_state.csv_data.append([
                        exercise,
                        st.session_state.counter,
                        st.session_state.stage,
                        st.session_state.pose_status,
                        st.session_state.l_angle,
                        st.session_state.r_angle
                    ])

                    # Show annotated image
                    frame_window.image(annotated, channels="RGB")
                except Exception:
                    # keep silent as original
                    frame_window.image(frame_rgb, channels="RGB")
            else:
                # No landmarks — show raw frame
                frame_window.image(frame_rgb, channels="RGB")

    # If camera widget present (or even if not), trigger a timed reload to get the next frame
    # We inject a tiny HTML+JS that reloads the page after REFRESH_MS only while run_cam is True.
    reload_js = f"""
    <script>
    // auto-reload for near-real-time capture
    setTimeout(() => {{
        try {{
            // append a timestamp to avoid aggressive caching
            const url = new URL(window.location.href);
            url.searchParams.set('_rt', Date.now());
            window.location.href = url.toString();
        }} catch (e) {{
            window.location.reload();
        }}
    }}, {REFRESH_MS});
    </script>
    """
    # height=0 keeps it invisible
    st.components.v1.html(reload_js, height=0)

else:
    # Camera not running — show placeholder or last frame
    if len(st.session_state.record_frames) > 0:
        last = st.session_state.record_frames[-1]
        last_rgb = cv2.cvtColor(last, cv2.COLOR_BGR2RGB)
        frame_window.image(last_rgb, channels="RGB")
    else:
        frame_window.image(np.zeros((480, 640, 3), dtype=np.uint8), channels="RGB")

# Right side metrics (always updated)
reps_box.markdown(f"### Reps: **{st.session_state.counter}**")
stage_box.markdown(f"### Stage: **{st.session_state.stage}**")
posture_box.markdown(f"### Posture: **{st.session_state.pose_status}**")
angle_box.markdown(f"### Angle: **{st.session_state.l_angle}°**")

# ---------------- Video download (unchanged) ----------------
if upload and len(st.session_state.record_frames) > 0:
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    height, width, _ = st.session_state.record_frames[0].shape
    writer = cv2.VideoWriter(temp_video.name, cv2.VideoWriter_fourcc(*"mp4v"), 20, (width, height))
    for frame in st.session_state.record_frames:
        writer.write(frame)
    writer.release()
    with open(temp_video.name, "rb") as video_file:
        st.download_button(label="Output Video", data=video_file, file_name="output_video.mp4", mime="video/mp4")
    st.session_state.record_frames = []

# ---------------- CSV download (unchanged) ----------------
if details and len(st.session_state.csv_data) > 0:
    df = pd.DataFrame(st.session_state.csv_data, columns=[
        "Exercise", "Reps", "Stage", "Posture Status", "Left Angle", "Right Angle"
    ])
    temp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(temp_csv.name, index=False)
    with open(temp_csv.name, "rb") as f:
        st.download_button(label="Download Details", data=f, file_name="pose_details.csv", mime="text/csv")
    st.session_state.csv_data = []
