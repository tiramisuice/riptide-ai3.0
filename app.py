"""Hands-raised detection demo with MediaPipe Pose Landmarker (Python version).
Uses the server-side MediaPipe Pose Landmarker Python API.
"""

import time
import base64
import uuid
import os
import requests
import json
import asyncio
from pathlib import Path
from datetime import datetime
import threading
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from split_video import analyze_photo_dataurl, summarize_drowning_likelihood
import queue

# Constants
HERE = Path(__file__).parent
OUTPUT_PHOTOS_DIR = HERE / "output_photos"
OUTPUT_PHOTOS_DIR.mkdir(exist_ok=True)
MODEL_DIR = HERE / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
MODEL_PATH = MODEL_DIR / "pose_landmarker_full.task"

# Global variables for thread control
snapshot_stop_event = threading.Event()
analysis_stop_event = threading.Event()
drowning_analysis_stop_event = threading.Event()
current_frame = None
frame_lock = threading.Lock()
snapshot_messages = []
snapshot_messages_lock = threading.Lock()
log_html_content = ""  # Global variable to store log HTML content
drowning_likelihood_result = "No data available"  # Global variable for drowning likelihood
drowning_likelihood_lock = threading.Lock()
snapshot_logs = []  # Global variable to store snapshot logs instead of session state
snapshot_logs_lock = threading.Lock()  # Lock for snapshot logs
alarm_triggered_permanently = False  # Flag for continuous alarm

# Queue for photo analysis tasks
analysis_queue = queue.Queue()

# Utility function to downscale an image for OpenAI analysis
def downscale_image_for_openai(image, max_width=640, max_height=480):
    """Downscale an image to a reasonable size for OpenAI API calls to reduce costs."""
    height, width = image.shape[:2]
    
    # If image is already smaller than max dimensions, return as is
    if width <= max_width and height <= max_height:
        return image
    
    # Calculate the scaling factor
    scale_width = max_width / width
    scale_height = max_height / height
    scale = min(scale_width, scale_height)
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return resized_image

# Utility function to convert frame to data URL
def frame_to_data_url(frame, for_openai=False):
    """Encode an OpenCV BGR frame (numpy array) into a JPEG Data URL."""
    if for_openai:
        # Downscale image if it's for OpenAI analysis
        frame = downscale_image_for_openai(frame)
    
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        raise RuntimeError("Failed to encode frame to JPEG")
    b64 = base64.b64encode(buffer).decode("ascii")
    data_url = f"data:image/jpeg;base64,{b64}"
    return data_url

# Download model if it doesn't exist
@st.cache_resource
def download_model():
    if not MODEL_PATH.exists():
        with st.spinner("Downloading pose detection model (this may take a moment)..."):
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success("Model downloaded successfully!")
    return MODEL_PATH

# Initialize MediaPipe Pose Detector
@st.cache_resource
def create_pose_detector():
    """Create and return a MediaPipe Pose Landmarker"""
    # First download the model if needed
    model_path = download_model()
    
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True,
        running_mode=vision.RunningMode.IMAGE
    )
    return vision.PoseLandmarker.create_from_options(options)

@st.cache_data
def get_alarm_audio():
    """Load alarm sound file"""
    alarm_file = HERE / "alarm.mp3"
    if alarm_file.exists():
        with open(alarm_file, "rb") as f:
            return f.read()
    return None

# Initialize session state variables
if 'last_alarm_time' not in st.session_state:
    st.session_state.last_alarm_time = 0
if 'alarm_triggered' not in st.session_state:
    st.session_state.alarm_triggered = False
if 'snapshot_thread' not in st.session_state:
    st.session_state.snapshot_thread = None
if 'analysis_thread' not in st.session_state:
    st.session_state.analysis_thread = None
if 'drowning_analysis_thread' not in st.session_state:
    st.session_state.drowning_analysis_thread = None
if 'taking_snapshots' not in st.session_state:
    st.session_state.taking_snapshots = False
if 'debug' not in st.session_state:
    st.session_state.debug = []

# Alarm settings in sidebar
st.sidebar.header("Detection Settings")
angle_threshold = st.sidebar.slider("Arm angle threshold (degrees)", 0, 180, 70, 5)
visibility_threshold = st.sidebar.slider("Landmark visibility threshold", 0.0, 1.0, 0.0, 0.05)
both_hands_required = st.sidebar.checkbox("Require both hands raised", value=False)

st.sidebar.header("Alarm Settings")
enable_alarm = st.sidebar.checkbox("Enable hands raised alarm", value=True)
alarm_cooldown = st.sidebar.slider("Alarm cooldown (seconds)", 1, 30, 5)
pose_duration_threshold = st.sidebar.slider("Hands raised duration (seconds)", 1, 10, 3)
percentage_threshold = st.sidebar.slider("% of time hands must be raised", 50, 100, 90, 5)

# Add alarm test section
st.sidebar.header("Alarm Test")
if st.sidebar.button("Test Alarm Sound"):
    # Get the alarm audio data
    test_alarm_audio = get_alarm_audio()
    if test_alarm_audio is not None:
        # Convert audio to base64
        audio_b64 = base64.b64encode(test_alarm_audio).decode()
        # Display audio player
        st.sidebar.markdown(
            f'<audio autoplay><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3"></audio>',
            unsafe_allow_html=True
        )
        st.sidebar.success("Alarm sound test triggered! You should hear the alarm.")
    else:
        st.sidebar.error("Alarm sound file not found. Please add 'alarm.mp3' to the app directory.")

# Video settings in sidebar
st.sidebar.header("Video Settings")
rotation_angle = st.sidebar.selectbox("Rotation angle", [0, 90, 180, 270], index=0)

# Get the alarm audio data
alarm_audio = get_alarm_audio()
if enable_alarm and alarm_audio is None:
    st.sidebar.error("Alarm sound file not found. Please add 'alarm.mp3' to the app directory.")

# Create placeholders for alarm UI elements
alarm_status = st.empty()
alarm_player = st.empty()

# Main content
st.title("Hands Raised Detection")
st.markdown("This app detects when people raise their hands and sends alerts.")

# Add Gen AI Recognition section
st.sidebar.header("Gen AI-Enhanced Recognition")
ai_col1, ai_col2 = st.sidebar.columns(2)
start_button = ai_col1.button("Start")
stop_button = ai_col2.button("Stop")

# Function to check if hands are raised based on pose landmarks
def is_hands_raised(landmarks):
    if not landmarks:
        return False
    
    try:
        # Indices for relevant landmarks
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_ELBOW = 13
        RIGHT_ELBOW = 14
        LEFT_WRIST = 15
        RIGHT_WRIST = 16
        
        # Extract landmark positions
        l_shoulder = landmarks[LEFT_SHOULDER]
        r_shoulder = landmarks[RIGHT_SHOULDER]
        l_wrist = landmarks[LEFT_WRIST]
        r_wrist = landmarks[RIGHT_WRIST]
        
        # Calculate visibility
        l_visibility = min(
            l_shoulder.visibility or 0,
            landmarks[LEFT_ELBOW].visibility or 0,
            l_wrist.visibility or 0
        )
        r_visibility = min(
            r_shoulder.visibility or 0,
            landmarks[RIGHT_ELBOW].visibility or 0,
            r_wrist.visibility or 0
        )
        
        # Check if visibility is above threshold
        if l_visibility < visibility_threshold and r_visibility < visibility_threshold:
            return False
        
        # Adjust position comparison based on rotation
        # Check if wrists are above shoulders
        l_wrist_above_shoulder = l_wrist.y < l_shoulder.y
        r_wrist_above_shoulder = r_wrist.y < r_shoulder.y
        
        # Determine if hands are raised based on configuration
        if both_hands_required:
            return l_wrist_above_shoulder and r_wrist_above_shoulder
        return l_wrist_above_shoulder or r_wrist_above_shoulder
    
    except Exception as e:
        st.error(f"Error detecting hands raised: {e}")
        return False

# Function to calculate arm angle
def calculate_arm_angle(shoulder, elbow):
    # Create vectors
    upper_arm_vector = [elbow.x - shoulder.x, elbow.y - shoulder.y]
    
    # Adjust vertical reference vector based on rotation
    if rotation_angle == 0:
        vertical_vector = [0, -1]  # Up is negative Y
    elif rotation_angle == 90:
        vertical_vector = [-1, 0]  # Up is negative X
    elif rotation_angle == 180:
        vertical_vector = [0, 1]   # Up is positive Y
    elif rotation_angle == 270:
        vertical_vector = [1, 0]   # Up is positive X
    else:
        vertical_vector = [0, -1]  # Default: Up is negative Y
    
    # Calculate magnitudes
    upper_arm_mag = np.sqrt(upper_arm_vector[0]**2 + upper_arm_vector[1]**2)
    
    # Guard against zero magnitude
    if upper_arm_mag == 0:
        return 180
    
    # Calculate normalized dot product
    dot_product = (upper_arm_vector[0] * vertical_vector[0] + 
                  upper_arm_vector[1] * vertical_vector[1]) / upper_arm_mag
    
    # Calculate angle in degrees
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0)) * (180 / np.pi)
    return angle

# Async function to process photo analysis tasks
async def process_analysis_queue():
    """Asynchronous worker that processes photo analysis tasks in parallel"""
    print("Analysis worker started")
    
    # Keep track of pending tasks
    tasks = []
    
    while not analysis_stop_event.is_set():
        try:
            # Check if there are items in the queue
            while not analysis_queue.empty():
                # Get an item from the queue
                item = analysis_queue.get_nowait()
                filename, data_url, readable_time = item
                
                # Start a new task to analyze the photo
                task = asyncio.create_task(
                    analyze_and_log(filename, data_url, readable_time)
                )
                tasks.append(task)
                
                # Clean up completed tasks
                tasks = [t for t in tasks if not t.done()]
                
            # Wait a short time before checking the queue again
            await asyncio.sleep(0.1)
            
        except queue.Empty:
            # Queue is empty, wait a bit before checking again
            await asyncio.sleep(0.5)
        except Exception as e:
            print(f"Error in analysis worker: {e}")
            await asyncio.sleep(1)
    
    # Wait for all remaining tasks to complete
    if tasks:
        print(f"Waiting for {len(tasks)} analysis tasks to complete...")
        await asyncio.gather(*tasks)
    
    print("Analysis worker stopped")

# Async function to analyze a photo and log the results
async def analyze_and_log(filename, data_url, readable_time):
    """Analyze a photo and log the results"""
    try:
        # Analyze the photo
        analysis_json = await analyze_photo_dataurl(data_url)
        
        # Format log message in the enhanced style
        log_message = (
            f"Snapshot: {filename} ({readable_time})<br>"
            f"{analysis_json}"
            f"<br>```<br>"
        )
        
        # Log to console
        print(f"Snapshot: {filename} ({readable_time})")
        print(f"{analysis_json}")
        print("```")
        
        # Store message in thread-safe way
        with snapshot_messages_lock:
            snapshot_messages.append(log_message)
            
    except Exception as e:
        print(f"Error analyzing photo {filename}: {e}")

# Function to start the analysis worker thread
def start_analysis_worker():
    """Start a thread that runs the async analysis worker"""
    def run_async_worker():
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run the analysis worker until stopped
        try:
            loop.run_until_complete(process_analysis_queue())
        finally:
            loop.close()
    
    # Create and start the thread
    thread = threading.Thread(target=run_async_worker, daemon=True)
    thread.start()
    return thread

# Async function to run drowning likelihood analysis
async def process_drowning_likelihood():
    """Async worker that periodically analyzes recent snapshots for drowning likelihood"""
    global drowning_likelihood_result
    
    print("Drowning likelihood analysis worker started")
    
    while not drowning_analysis_stop_event.is_set():
        try:
            # Get the 20 most recent snapshots (which are now at the beginning of the list)
            snapshots_to_analyze = []
            with snapshot_logs_lock:
                # Get up to 20 most recent snapshots from the global variable
                snapshots_to_analyze = snapshot_logs[:20]
            
            if snapshots_to_analyze:
                print(f"Analyzing {len(snapshots_to_analyze)} snapshots for drowning likelihood")
                
                # Call the summarize_drowning_likelihood function
                result = await summarize_drowning_likelihood(snapshots_to_analyze)
                
                # Store the result
                with drowning_likelihood_lock:
                    drowning_likelihood_result = result
                
                # Log to console
                print(f"Drowning likelihood result: {result}")
            else:
                print("No snapshots available for drowning likelihood analysis")
            
            # Wait for 3 seconds before the next analysis
            for _ in range(5):  # 5 x 0.1s = 0.5s
                await asyncio.sleep(0.1)
                if drowning_analysis_stop_event.is_set():
                    break
                
        except Exception as e:
            print(f"Error in drowning likelihood analysis: {e}")
            await asyncio.sleep(1)
    
    print("Drowning likelihood analysis worker stopped")

# Function to start the drowning likelihood analysis thread
def start_drowning_analysis_worker():
    """Start a thread that runs the async drowning likelihood analysis worker"""
    def run_async_worker():
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run the drowning likelihood analysis worker until stopped
        try:
            loop.run_until_complete(process_drowning_likelihood())
        finally:
            loop.close()
    
    # Create and start the thread
    thread = threading.Thread(target=run_async_worker, daemon=True)
    thread.start()
    return thread

# Function to take snapshots at regular intervals
def snapshot_thread_function():
    print("Snapshot thread started")
    
    while not snapshot_stop_event.is_set():
        try:
            # Get current frame safely
            with frame_lock:
                frame = current_frame
            
            # Check stop event again to exit immediately if needed
            if snapshot_stop_event.is_set():
                break
                
            if frame is not None:
                # Generate a unique filename with timestamp and UUID
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                unique_id = str(uuid.uuid4())[:8]
                filename = f"{timestamp}_{unique_id}.jpg"
                filepath = OUTPUT_PHOTOS_DIR / filename
                
                # Save the frame
                cv2.imwrite(str(filepath), frame)
                
                # Format readable time from filename
                readable_time = datetime.strptime(timestamp, "%Y%m%d-%H%M%S").strftime("%I:%M:%S %p")
                
                # Convert frame to data URL for analysis - downscale for OpenAI
                data_url = frame_to_data_url(frame, for_openai=True)
                
                # Add to analysis queue instead of analyzing directly
                analysis_queue.put((filename, data_url, readable_time))
                
                # Log that we've queued the snapshot (optional)
                print(f"Queued snapshot for analysis: {filename}")
                
        except Exception as e:
            print(f"Error saving snapshot: {e}")
        
        # Check stop event again after each iteration and before sleep
        if snapshot_stop_event.is_set():
            break
            
        # Wait with a shorter interval, checking stop event more frequently
        for _ in range(5):  # 5 x 0.1s = 0.5s
            time.sleep(0.1)
            if snapshot_stop_event.is_set():
                break
    
    print("Snapshot thread stopped")

# Main webcam loop
def main():
    global current_frame, log_html_content, drowning_likelihood_result, alarm_triggered_permanently
    
    pose_detector = create_pose_detector()
    
    # Window for tracking hands raised percentage
    window_size = 30  # Frames to consider (should cover roughly 3 seconds at 10fps)
    hands_raised_history = [False] * window_size
    history_index = 0
    hands_raised_start_time = 0
    hands_raised_state = False
    
    # Metrics for display
    fps_counter = 0
    fps_start_time = time.time()
    
    # Setup webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam. Please check your camera connection.")
        return
    
    while True:
        success, frame = cap.read()
        if not success:
            st.error("Failed to read from webcam")
            time.sleep(1)
            continue
        
        # Apply rotation if needed
        if rotation_angle == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_angle == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation_angle == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Mirror horizontally
        frame = cv2.flip(frame, 1)
        
        # Store frame for snapshots with thread safety
        with frame_lock:
            current_frame = frame.copy()
        
        # Process frame with MediaPipe
        processed_frame, current_hands_raised = process_frame(frame, pose_detector)
        
        # Update hands raised history
        hands_raised_history[history_index] = current_hands_raised
        history_index = (history_index + 1) % window_size
        
        # Calculate hands raised percentage
        hands_raised_percentage = (sum(hands_raised_history) / window_size) * 100
        
        # Handle hands raised state based on percentage threshold
        if hands_raised_percentage >= percentage_threshold:
            # Track duration of hands raised state
            if hands_raised_start_time == 0:
                hands_raised_start_time = time.time()
            
            # Check if hands raised long enough to trigger alert
            duration = time.time() - hands_raised_start_time
            if duration >= pose_duration_threshold and not hands_raised_state:
                hands_raised_state = True
                # Trigger alarm
                current_time = time.time()
                if enable_alarm and alarm_audio is not None:
                    if (current_time - st.session_state.last_alarm_time) > alarm_cooldown:
                        st.session_state.last_alarm_time = current_time
                        st.session_state.alarm_triggered = True
                        alarm_triggered_permanently = True  # Set permanent flag
                        with alarm_status:
                            st.warning("⚠️ ALERT: Hands raised detected for extended period!")
        else:
            # Reset when hands raised percentage drops below threshold
            if hands_raised_start_time > 0:
                hands_raised_start_time = 0
                if hands_raised_state:
                    hands_raised_state = False
                    # Don't reset alarm status anymore
                    # Only clear the temporary UI elements but keep alarm_triggered_permanently = True
        
        # Play continuous alarm if triggered
        if alarm_triggered_permanently and alarm_audio:
            audio_b64 = base64.b64encode(alarm_audio).decode()
            with alarm_player:
                st.markdown(
                    f'<audio autoplay loop><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3"></audio>',
                    unsafe_allow_html=True
                )
            with alarm_status:
                st.warning("⚠️ ALERT: Alarm triggered! Refresh page to reset.")

        # Create containers for UI elements if they don't exist yet
        if 'drowning_likelihood_container' not in locals():
            drowning_likelihood_container = st.empty()
            
        # Display frame
        video_frame.image(processed_frame, channels="BGR", use_container_width=True)
        
        # Calculate and display FPS
        fps_counter += 1
        elapsed = time.time() - fps_start_time
        if elapsed >= 1.0:
            fps = fps_counter / elapsed
            fps_placeholder.text(f"FPS: {fps:.1f}")
            fps_counter = 0
            fps_start_time = time.time()
        
        # Debug info
        debug_info = {
            "hands_raised": current_hands_raised,
            "percentage": f"{hands_raised_percentage:.1f}%",
            "threshold": f"{percentage_threshold}%",
            "snapshot_enabled": st.session_state.taking_snapshots
        }
        debug_text = "\n".join([f"{key}: {value}" for key, value in debug_info.items()])
        debug_placeholder.text(debug_text)
        
        # Display any snapshot messages
        with snapshot_messages_lock:
            if snapshot_messages:
                for msg in snapshot_messages:
                    # Prepend to global snapshot logs with thread safety
                    with snapshot_logs_lock:
                        snapshot_logs.insert(0, msg)
                snapshot_messages.clear()
        
        # Update drowning likelihood display
        with drowning_likelihood_lock:
            current_likelihood_result = drowning_likelihood_result
            
            # Check if "alert" is in the drowning likelihood result (case-insensitive)
            if "alert" in current_likelihood_result.lower():
                # Trigger permanent alarm
                alarm_triggered_permanently = True
                print("ALERT detected in drowning likelihood result - triggering permanent alarm")
                with alarm_status:
                    st.warning("⚠️ ALERT: Drowning risk detected! Refresh page to reset.")
            
            drowning_html = f"""
                <div style="padding: 10px; border: 1px solid #333; background-color: #f8f9fa !important; margin-bottom: 10px; color: #000000 !important; box-shadow: 0 1px 2px rgba(0,0,0,0.1);">
                    <h3 style="margin: 0; color: #0066cc !important; font-weight: bold;">Drowning Likelihood</h3>
                    <p style="color: #000000 !important; margin-top: 5px; font-size: 16px;">{current_likelihood_result}</p>
                </div>
            """
        drowning_likelihood_container.markdown(drowning_html, unsafe_allow_html=True)
        
        # Update log content HTML with better formatting
        log_html_content = "<div class='snapshot-log-box'>"
        with snapshot_logs_lock:
            for log in snapshot_logs:
                # Apply proper formatting to the log
                log = log.replace("Snapshot:", "<span class='snapshot-title'>Snapshot:</span>")
                log = log.replace("```json<br>", "<pre><code>")
                log = log.replace("```<br>", "</code></pre>")
                log_html_content += f"{log}<br>"
        log_html_content += "</div>"
        
        # Display logs in the scrollable div (using the same container)
        snapshot_log_container.markdown(log_html_content, unsafe_allow_html=True)
        
        # Short delay to reduce CPU usage
        time.sleep(0.01)

# Handle start/stop buttons for snapshot collection
if start_button and not st.session_state.taking_snapshots:
    # Create output directory if it doesn't exist
    OUTPUT_PHOTOS_DIR.mkdir(exist_ok=True)
    
    # Reset the stop events
    snapshot_stop_event.clear()
    analysis_stop_event.clear()
    drowning_analysis_stop_event.clear()
    
    # Start analysis worker thread
    st.session_state.analysis_thread = start_analysis_worker()
    
    # Start snapshot thread
    st.session_state.snapshot_thread = threading.Thread(
        target=snapshot_thread_function, daemon=True
    )
    st.session_state.snapshot_thread.start()
    
    # Start drowning likelihood analysis thread
    st.session_state.drowning_analysis_thread = start_drowning_analysis_worker()
    
    # Update UI
    st.session_state.taking_snapshots = True
    st.sidebar.success("AI recognition started! Taking snapshots every 0.5 seconds.")
    print("Started taking snapshots. Files will be saved to:", OUTPUT_PHOTOS_DIR)
    st.session_state.debug.append(f"Started snapshots at {datetime.now()}")

if stop_button and st.session_state.taking_snapshots:
    # Signal threads to stop
    snapshot_stop_event.set()
    drowning_analysis_stop_event.set()
    
    # Wait for snapshot thread to terminate with timeout
    if st.session_state.snapshot_thread:
        st.sidebar.info("Stopping snapshots...")
        st.session_state.snapshot_thread.join(timeout=1)
        if st.session_state.snapshot_thread.is_alive():
            st.session_state.debug.append("Snapshot thread is taking longer to stop than expected")
    
    # Wait for any queued analyses to complete (with timeout)
    st.sidebar.info("Waiting for analysis queue to empty...")
    timeout = 5  # 5 seconds timeout for queue to drain
    start_time = time.time()
    while not analysis_queue.empty() and (time.time() - start_time) < timeout:
        time.sleep(0.1)
    
    # Signal analysis thread to stop
    analysis_stop_event.set()
    
    # Wait for analysis thread to terminate (with timeout)
    if st.session_state.analysis_thread:
        st.session_state.analysis_thread.join(timeout=2)
    
    # Wait for drowning analysis thread to terminate (with timeout)
    if st.session_state.drowning_analysis_thread:
        st.session_state.drowning_analysis_thread.join(timeout=2)
    
    # Update UI
    st.session_state.taking_snapshots = False
    st.sidebar.success("AI recognition stopped successfully.")
    print("Stopped taking snapshots.")
    st.session_state.debug.append(f"Stopped snapshots at {datetime.now()}")

# Setup webcam capture
video_frame = st.empty()
fps_placeholder = st.empty()
debug_placeholder = st.empty()

# Add CSS for scrollable log container (outside main loop)
st.markdown("""
    <style>
        .snapshot-log-box {
            height: 400px;
            overflow-y: scroll;
            border: 1px solid #ddd;
            padding: 10px;
            font-family: monospace;
            background-color: #f8f9fa;
            color: black !important;
        }
        .snapshot-log-box pre {
            margin: 0;
            white-space: pre-wrap;
        }
        .snapshot-log-box code {
            color: black;
            background-color: #f1f1f1;
            padding: 2px 4px;
            border-radius: 3px;
            display: block;
            margin: 5px 0;
            overflow-x: auto;
        }
        .snapshot-title {
            font-weight: bold;
            color: #0066cc;
        }
        .snapshot-frame {
            color: #666;
            font-style: italic;
        }
    </style>
""", unsafe_allow_html=True)

# Create containers for UI elements
drowning_likelihood_container = st.empty()
snapshot_log_container = st.empty()

# Function to process frame with MediaPipe
def process_frame(frame, pose_detector):
    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Detect pose landmarks
    detection_result = pose_detector.detect(mp_image)
    
    # Draw landmarks on the frame
    annotated_image = np.copy(frame)
    
    # Check if any hands are raised
    hands_raised = False
    
    if detection_result.pose_landmarks:
        # Draw landmarks on the image
        height, width, _ = annotated_image.shape
        pose_landmarks_list = detection_result.pose_landmarks
        
        # Initialize drawing utilities
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        
        # Convert normalized landmarks to pixel coordinates
        for idx, pose_landmarks in enumerate(pose_landmarks_list):
            # Check if this person has hands raised
            if is_hands_raised(pose_landmarks):
                hands_raised = True
            
            # Draw the pose landmarks manually since we can't use PoseLandmarksProtobuf
            for landmark_idx, landmark in enumerate(pose_landmarks):
                # Convert normalized coordinates to pixel values
                landmark_px = mp_drawing._normalized_to_pixel_coordinates(
                    landmark.x, landmark.y, width, height)
                
                if landmark_px:
                    cv2.circle(
                        annotated_image, 
                        landmark_px, 
                        5,  # radius
                        (0, 255, 0),  # color (green)
                        -1  # filled circle
                    )
            
            # Draw connections between landmarks
            connections = mp_pose.POSE_CONNECTIONS
            for connection in connections:
                start_idx, end_idx = connection
                if (start_idx < len(pose_landmarks) and end_idx < len(pose_landmarks)):
                    start = pose_landmarks[start_idx]
                    end = pose_landmarks[end_idx]
                    
                    # Check if landmarks are visible
                    if ((start.visibility or 0) > visibility_threshold and 
                        (end.visibility or 0) > visibility_threshold):
                        
                        # Convert to pixel coordinates
                        start_px = mp_drawing._normalized_to_pixel_coordinates(
                            start.x, start.y, width, height)
                        end_px = mp_drawing._normalized_to_pixel_coordinates(
                            end.x, end.y, width, height)
                        
                        if start_px and end_px:
                            cv2.line(
                                annotated_image, 
                                start_px, 
                                end_px, 
                                (0, 0, 255),  # color (blue)
                                2  # thickness
                            )
            
            # Draw hands raised indicator if detected
            if hands_raised:
                cv2.putText(
                    annotated_image,
                    "HANDS RAISED!",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )
    
    return annotated_image, hands_raised

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {e}")

# About section
st.markdown("""
---
## About
This demo uses MediaPipe Pose Detection Python API for detecting hand raises.
When the AI recognition is enabled, it saves snapshots every 0.5 seconds for further analysis.
""")

# Show the last 5 debug messages in an expandable section
with st.expander("Debug Information", expanded=False):
    for msg in st.session_state.debug[-5:]:
        st.text(msg)
    if st.button("Clear Debug Log"):
        st.session_state.debug = []
