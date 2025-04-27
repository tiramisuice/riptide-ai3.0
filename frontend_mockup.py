"""Hands-raised detection demo with MediaPipe Pose Landmarker.
Uses the client-side MediaPipe Pose Landmarker for efficiency.
"""

import time
import base64
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

HERE = Path(__file__).parent


@st.cache_data
def get_alarm_audio():
    """Load alarm sound file"""
    alarm_file = HERE / "alarm.mp3"
    if alarm_file.exists():
        with open(alarm_file, "rb") as f:
            return f.read()
    return None


# Alarm settings in sidebar
st.sidebar.header("Detection Settings")
angle_threshold = st.sidebar.slider("Arm angle threshold (degrees)", 0, 180, 70, 5)
visibility_threshold = st.sidebar.slider(
    "Landmark visibility threshold", 0.0, 1.0, 0.0, 0.05
)
both_hands_required = st.sidebar.checkbox("Require both hands raised", value=False)

st.sidebar.header("Alarm Settings")
enable_alarm = st.sidebar.checkbox("Enable hands raised alarm", value=True)
alarm_cooldown = st.sidebar.slider("Alarm cooldown (seconds)", 1, 30, 5)
pose_duration_threshold = st.sidebar.slider("Hands raised duration (seconds)", 1, 10, 3)
percentage_threshold = st.sidebar.slider(
    "% of time hands must be raised", 50, 100, 90, 5
)

# Video settings in sidebar
st.sidebar.header("Video Settings")
rotation_angle = st.sidebar.selectbox("Rotation angle", [0, 90, 180, 270], index=0)

# Get the alarm audio data
alarm_audio = get_alarm_audio()
if enable_alarm and alarm_audio is None:
    st.sidebar.error(
        "Alarm sound file not found. Please add 'alarm.mp3' to the app directory."
    )

# Initialize session state for alarm
if "last_alarm_time" not in st.session_state:
    st.session_state.last_alarm_time = 0
if "alarm_triggered" not in st.session_state:
    st.session_state.alarm_triggered = False

# Create placeholders for alarm UI elements
alarm_status = st.empty()
alarm_player = st.empty()

# =====================
# PLACEHOLDER FOR VIDEO
# =====================

# Debug output for easy troubleshooting
if "debug" not in st.session_state:
    st.session_state.debug = []

# Initialize status display
status_placeholder = st.empty()
if st.session_state.alarm_triggered:
    status_placeholder.warning("⚠️ Previous alert is still active")

# Show the last 3 debug messages in an expandable section
with st.expander("Debug Information", expanded=False):
    for msg in st.session_state.debug[-3:]:
        st.text(msg)
    if st.button("Clear Debug Log"):
        st.session_state.debug = []

# About section
st.markdown(
    """
---
## About
This demo uses MediaPipe Pose Detection running in your browser for maximum performance.
The pose detection runs at 25-30 FPS on most modern devices.
"""
)
