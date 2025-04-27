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
visibility_threshold = st.sidebar.slider("Landmark visibility threshold", 0.0, 1.0, 0.0, 0.05)
both_hands_required = st.sidebar.checkbox("Require both hands raised", value=False)

st.sidebar.header("Alarm Settings")
enable_alarm = st.sidebar.checkbox("Enable hands raised alarm", value=True)
alarm_cooldown = st.sidebar.slider("Alarm cooldown (seconds)", 1, 30, 5)
pose_duration_threshold = st.sidebar.slider("Hands raised duration (seconds)", 1, 10, 3)
percentage_threshold = st.sidebar.slider("% of time hands must be raised", 50, 100, 90, 5)

# Video settings in sidebar
st.sidebar.header("Video Settings")
rotation_angle = st.sidebar.selectbox("Rotation angle", [0, 90, 180, 270], index=0)

# Get the alarm audio data
alarm_audio = get_alarm_audio()
if enable_alarm and alarm_audio is None:
    st.sidebar.error("Alarm sound file not found. Please add 'alarm.mp3' to the app directory.")

# Initialize session state for alarm
if 'last_alarm_time' not in st.session_state:
    st.session_state.last_alarm_time = 0
if 'alarm_triggered' not in st.session_state:
    st.session_state.alarm_triggered = False

# Create placeholders for alarm UI elements
alarm_status = st.empty()
alarm_player = st.empty()

# Main content
st.title("Hands Raised Detection")
st.markdown("This app detects when people raise their hands and sends alerts.")

# Create the custom component with the MediaPipe code
pose_detector_html = f"""
<!DOCTYPE html>
<html>
<head>
  <style>
    #liveView {{
      position: relative;
      width: 100%;
      overflow: hidden;
    }}
    #webcam {{
      width: 100%;
      height: auto;
      transform: rotateY(180deg) rotate({rotation_angle}deg);
    }}
    #output_canvas {{
      position: absolute;
      left: 0;
      top: 0;
      width: 100%;
      height: 100%;
      transform: rotateY(180deg) rotate({rotation_angle}deg);
    }}
    #camera-controls {{
      margin: 10px 0;
      padding: 5px;
      background-color: rgba(0,0,0,0.1);
      border-radius: 5px;
    }}
    #fps-counter {{
      position: absolute;
      top: 10px;
      right: 10px;
      background-color: rgba(0,0,0,0.5);
      color: white;
      padding: 5px;
      border-radius: 5px;
      font-family: Arial, sans-serif;
    }}
    #debug-overlay {{
      position: absolute;
      bottom: 10px;
      left: 10px;
      background-color: rgba(0,0,0,0.5);
      color: white;
      padding: 8px;
      border-radius: 5px;
      font-family: Arial, sans-serif;
      font-size: 12px;
      max-width: 350px;
      white-space: pre-wrap;
    }}
  </style>
</head>
<body>
  <div id="camera-controls">
    <select id="camera-select">
      <option value="">Loading cameras...</option>
    </select>
    <button id="switch-camera">Switch Camera</button>
  </div>
  <div id="liveView">
    <video id="webcam" autoplay playsinline></video>
    <canvas id="output_canvas"></canvas>
    <div id="fps-counter">FPS: 0</div>
    <div id="debug-overlay"></div>
  </div>
  
  <script>
    // Store Streamlit configuration in global variables
    window.poseConfig = {{
      angleThreshold: {angle_threshold},
      visibilityThreshold: {visibility_threshold},
      bothHandsRequired: {str(both_hands_required).lower()},
      durationThreshold: {pose_duration_threshold * 1000},  // Convert to milliseconds
      percentageThreshold: {percentage_threshold},  // Percentage of time hands must be raised
      rotationAngle: {rotation_angle}  // Rotation angle in degrees
    }};
  </script>
  
  <script type="module">
    // Import MediaPipe modules
    import {{
      PoseLandmarker,
      FilesetResolver,
      DrawingUtils
    }} from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";
    
    // Elements
    const videoElement = document.getElementById('webcam');
    const canvasElement = document.getElementById('output_canvas');
    const canvasCtx = canvasElement.getContext('2d');
    const fpsCounter = document.getElementById('fps-counter');
    const debugOverlay = document.getElementById('debug-overlay');
    const cameraSelect = document.getElementById('camera-select');
    const switchCameraButton = document.getElementById('switch-camera');
    
    // State variables
    let poseLandmarker = null;
    let runningMode = "VIDEO";
    let webcamRunning = false;
    let lastVideoTime = -1;
    let frameCount = 0;
    let startTime = Date.now();
    let handsRaised = false;
    let handsRaisedStartTime = 0;
    let debugInfo = {{}};
    let currentStream = null;
    let availableCameras = [];
    let selectedCameraId = null;
    
    // Window for tracking hands raised percentage
    const windowSize = 30; // Frames to consider (should cover roughly 3 seconds at 30fps)
    const handsRaisedHistory = Array(windowSize).fill(false);
    let historyIndex = 0;
    
    // Access config from window object
    const config = window.poseConfig;
    
    // Initialize camera selection UI
    async function initCameraSelection() {{
      try {{
        const devices = await navigator.mediaDevices.enumerateDevices();
        availableCameras = devices.filter(device => device.kind === 'videoinput');
        
        // Clear existing options
        cameraSelect.innerHTML = '';
        
        // Add options for each camera
        availableCameras.forEach((camera, index) => {{
          const option = document.createElement('option');
          option.value = camera.deviceId;
          option.text = camera.label || `Camera ${{index + 1}}`;
          cameraSelect.appendChild(option);
        }});
        
        if (availableCameras.length > 0) {{
          selectedCameraId = availableCameras[0].deviceId;
        }}
        
        // Enable switch camera button if more than one camera
        switchCameraButton.disabled = availableCameras.length <= 1;
      }} catch (error) {{
        console.error("Error enumerating devices:", error);
        debugOverlay.textContent = `Error listing cameras: ${{error.message}}`;
      }}
    }}
    
    // Handle camera selection change
    cameraSelect.addEventListener('change', () => {{
      selectedCameraId = cameraSelect.value;
      if (webcamRunning) {{
        // Stop current stream
        if (currentStream) {{
          currentStream.getTracks().forEach(track => track.stop());
        }}
        // Start new stream with selected camera
        enableCam();
      }}
    }});
    
    // Handle switch camera button
    switchCameraButton.addEventListener('click', () => {{
      if (availableCameras.length <= 1) return;
      
      // Find current camera index
      const currentIndex = availableCameras.findIndex(camera => camera.deviceId === selectedCameraId);
      // Get next camera index (cycle through available cameras)
      const nextIndex = (currentIndex + 1) % availableCameras.length;
      // Update selected camera
      selectedCameraId = availableCameras[nextIndex].deviceId;
      cameraSelect.value = selectedCameraId;
      
      // Restart webcam with new camera
      if (webcamRunning) {{
        // Stop current stream
        if (currentStream) {{
          currentStream.getTracks().forEach(track => track.stop());
        }}
        // Start new stream with selected camera
        enableCam();
      }}
    }});
    
    // Function to check if hands are raised
    function isHandsRaised(landmarks) {{
      if (!landmarks || landmarks.length === 0) {{
        return false;
      }}
      
      try {{
        // Indices for relevant landmarks
        const LEFT_SHOULDER = 11;
        const RIGHT_SHOULDER = 12;
        const LEFT_ELBOW = 13;
        const RIGHT_ELBOW = 14;
        const LEFT_WRIST = 15;
        const RIGHT_WRIST = 16;
        
        // Extract landmark positions
        const lShoulder = landmarks[LEFT_SHOULDER];
        const rShoulder = landmarks[RIGHT_SHOULDER];
        const lElbow = landmarks[LEFT_ELBOW];
        const rElbow = landmarks[RIGHT_ELBOW];
        const lWrist = landmarks[LEFT_WRIST];
        const rWrist = landmarks[RIGHT_WRIST];
        
        // Check visibility of landmarks - only log them and don't filter out
        const lShoulderVis = lShoulder.visibility || 0;
        const rShoulderVis = rShoulder.visibility || 0;
        const lElbowVis = lElbow.visibility || 0;
        const rElbowVis = rElbow.visibility || 0;
        const lWristVis = lWrist.visibility || 0;
        const rWristVis = rWrist.visibility || 0;
        
        const lVisibility = Math.min(lShoulderVis, lElbowVis, lWristVis);
        const rVisibility = Math.min(rShoulderVis, rElbowVis, rWristVis);
        
        // Update debug info
        debugInfo.landmarks = true;
        debugInfo.lVisibility = lVisibility.toFixed(2);
        debugInfo.rVisibility = rVisibility.toFixed(2);
        
        // Adjust position comparison based on rotation
        function isWristAboveShoulder(wrist, shoulder) {{
          return wrist.y < shoulder.y;
        }}
        
        // Simple position-based detection with rotation adjustment
        const lWristAboveShoulder = isWristAboveShoulder(lWrist, lShoulder);
        const rWristAboveShoulder = isWristAboveShoulder(rWrist, rShoulder);
        
        debugInfo.lWristAboveShoulder = lWristAboveShoulder;
        debugInfo.rWristAboveShoulder = rWristAboveShoulder;
        
        // Also track absolute positions for debugging
        debugInfo.lWristY = lWrist.y.toFixed(2);
        debugInfo.lShoulderY = lShoulder.y.toFixed(2); 
        debugInfo.rWristY = rWrist.y.toFixed(2);
        debugInfo.rShoulderY = rShoulder.y.toFixed(2);
        
        // Calculate angle between upper arm and vertical, adjusting for rotation
        function calculateArmAngle(shoulder, elbow) {{
          // Create vectors
          let upperArmVector, verticalVector;
          
          // Always create the vector from shoulder to elbow
          upperArmVector = [elbow.x - shoulder.x, elbow.y - shoulder.y];
          
          // Adjust vertical reference vector based on rotation
          if (config.rotationAngle === 0) {{
            verticalVector = [0, -1]; // Up is negative Y
          }} else if (config.rotationAngle === 90) {{
            verticalVector = [-1, 0]; // Up is negative X
          }} else if (config.rotationAngle === 180) {{
            verticalVector = [0, 1]; // Up is positive Y
          }} else if (config.rotationAngle === 270) {{
            verticalVector = [1, 0]; // Up is positive X
          }} else {{
            verticalVector = [0, -1]; // Default: Up is negative Y
          }}
          
          // Calculate magnitudes
          const upperArmMag = Math.sqrt(upperArmVector[0]**2 + upperArmVector[1]**2);
          const verticalMag = 1;
          
          // Guard against zero magnitude
          if (upperArmMag === 0) return 180;
          
          // Calculate normalized dot product
          const dotProduct = (upperArmVector[0] * verticalVector[0] + 
                              upperArmVector[1] * verticalVector[1]) / 
                              (upperArmMag * verticalMag);
          
          // Calculate angle in degrees
          const angle = Math.acos(Math.max(-1, Math.min(1, dotProduct))) * (180 / Math.PI);
          return angle;
        }}
        
        // Calculate angles
        const lAngle = calculateArmAngle(lShoulder, lElbow);
        const rAngle = calculateArmAngle(rShoulder, rElbow);
        
        // Update debug info with angles
        debugInfo.lAngle = lAngle.toFixed(1);
        debugInfo.rAngle = rAngle.toFixed(1);
        
        // Make hand raised detection more lenient - we just need wrists above shoulders
        // No strict angle requirement - just check if wrist is higher than shoulder
        // This increases detection sensitivity to avoid false negatives
        const lHandRaised = lWristAboveShoulder;
        const rHandRaised = rWristAboveShoulder;
        
        debugInfo.lHandRaised = lHandRaised;
        debugInfo.rHandRaised = rHandRaised;
        
        // Determine if hands are raised based on configuration
        const handsRaised = config.bothHandsRequired ? 
                           (lHandRaised && rHandRaised) : 
                           (lHandRaised || rHandRaised);
        
        return handsRaised;
      }} catch (error) {{
        console.error("Error detecting hands raised:", error);
        debugInfo.error = error.message;
        return false;
      }}
    }}
    
    // Function to calculate percentage of frames with hands raised
    function calculateHandsRaisedPercentage() {{
      const trueCount = handsRaisedHistory.filter(value => value).length;
      return (trueCount / windowSize) * 100;
    }}
    
    // Send data to Streamlit
    function sendToStreamlit(data) {{
      if (window.Streamlit) {{
        window.Streamlit.setComponentValue(data);
      }} else {{
        console.log("Streamlit not found, would send:", data);
      }}
    }}
    
    // Create the pose landmarker
    async function createPoseLandmarker() {{
      try {{
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
        );
        
        poseLandmarker = await PoseLandmarker.createFromOptions(vision, {{
          baseOptions: {{
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
            delegate: "GPU"
          }},
          runningMode: runningMode,
          numPoses: 2  // Detect up to 2 people
        }});
        
        console.log("Pose landmarker created successfully");
        
        // Initialize camera selection first
        await initCameraSelection();
        
        // Then enable webcam
        enableCam();
      }} catch (error) {{
        console.error("Error creating pose landmarker:", error);
        debugOverlay.textContent = `Error creating pose landmarker: ${{error.message}}`;
      }}
    }}
    
    // Enable the webcam
    function enableCam() {{
      if (!poseLandmarker) {{
        console.log("Wait for poseLandmarker to load!");
        return;
      }}
      
      console.log("Enabling webcam...");
      
      // Stop current stream if running
      if (currentStream) {{
        currentStream.getTracks().forEach(track => track.stop());
      }}
      
      // Request webcam access
      const constraints = {{
        video: selectedCameraId ? {{deviceId: selectedCameraId}} : true
      }};
      
      navigator.mediaDevices.getUserMedia(constraints)
        .then((stream) => {{
          videoElement.srcObject = stream;
          currentStream = stream;
          videoElement.addEventListener("loadeddata", predictWebcam);
          webcamRunning = true;
        }})
        .catch((err) => {{
          console.error("Error getting webcam access:", err);
          debugOverlay.textContent = `Error getting webcam access: ${{err.message}}`;
        }});
    }}
    
    // Main prediction loop for webcam
    async function predictWebcam() {{
      // Reset debug info
      debugInfo = {{}};
      
      // Calculate FPS
      frameCount++;
      const elapsed = (Date.now() - startTime) / 1000;
      if (elapsed >= 1) {{
        const fps = frameCount / elapsed;
        fpsCounter.textContent = `FPS: ${{fps.toFixed(1)}}`;
        debugInfo.fps = fps.toFixed(1);
        frameCount = 0;
        startTime = Date.now();
      }}
      
      // Set canvas dimensions to match video
      if (videoElement.videoWidth) {{
        canvasElement.width = videoElement.videoWidth;
        canvasElement.height = videoElement.videoHeight;
      }}
      
      // Only run detection when video is playing and time has changed
      if (videoElement.currentTime !== lastVideoTime) {{
        lastVideoTime = videoElement.currentTime;
        
        try {{
          // Run pose detection
          let startTimeMs = performance.now();
          const results = poseLandmarker.detectForVideo(videoElement, startTimeMs);
          
          // Clear canvas and prepare for drawing
          canvasCtx.save();
          canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
          
          // Flag to track if any person has hands raised
          let anyHandsRaised = false;
          
          // Process detection results
          if (results.landmarks && results.landmarks.length > 0) {{
            debugInfo.peopleDetected = results.landmarks.length;
            
            // Process each detected person
            for (const landmarks of results.landmarks) {{
              // Create drawing utils for this frame
              const drawingUtils = new DrawingUtils(canvasCtx);
              
              // Draw the pose landmarks
              drawingUtils.drawLandmarks(landmarks, {{
                radius: (data) => DrawingUtils.lerp(data.from?.z || 0, -0.15, 0.1, 5, 1)
              }});
              drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS);
              
              // Check for hands raised
              if (isHandsRaised(landmarks)) {{
                anyHandsRaised = true;
                
                // No need to check other people if one has hands raised
                break;
              }}
            }}
          }} else {{
            debugInfo.noPeopleDetected = true;
          }}
          
          // Update hands raised history window
          handsRaisedHistory[historyIndex] = anyHandsRaised;
          historyIndex = (historyIndex + 1) % windowSize;
          
          // Calculate percentage of time hands were raised
          const handsRaisedPercentage = calculateHandsRaisedPercentage();
          
          // Update debug info with key metrics
          debugInfo.anyHandsRaised = anyHandsRaised;
          debugInfo.handsRaisedPercentage = handsRaisedPercentage.toFixed(1);
          debugInfo.threshold = config.percentageThreshold;
          debugInfo.angleThreshold = config.angleThreshold;
          debugInfo.rotation = config.rotationAngle;
          
          // Display debug information
          const debugText = Object.entries(debugInfo)
            .map(([key, value]) => `${{key}}: ${{value}}`)
            .join('\\n');
          debugOverlay.textContent = debugText;
          
          // Handle hands raised state based on percentage threshold
          if (handsRaisedPercentage >= config.percentageThreshold) {{
            // Show visual indicator
            canvasCtx.font = "30px Arial";
            canvasCtx.fillStyle = "red";
            canvasCtx.fillText("HANDS RAISED!", 10, 40);
            
            // Track duration of hands raised state
            if (handsRaisedStartTime === 0) {{
              handsRaisedStartTime = Date.now();
            }}
            
            // Check if hands raised long enough to send alert
            const duration = Date.now() - handsRaisedStartTime;
            if (duration >= config.durationThreshold && !handsRaised) {{
              handsRaised = true;
              // Send message to Streamlit
              sendToStreamlit({{ hands_raised: true }});
            }}
          }} else {{
            // Reset when hands raised percentage drops below threshold
            if (handsRaisedStartTime > 0) {{
              handsRaisedStartTime = 0;
              if (handsRaised) {{
                handsRaised = false;
                // Send message to Streamlit
                sendToStreamlit({{ hands_raised: false }});
              }}
            }}
          }}
          
          canvasCtx.restore();
        }} catch (error) {{
          console.error("Error in prediction:", error);
          debugInfo.predictionError = error.message;
          debugOverlay.textContent = `Error in prediction: ${{error.message}}`;
        }}
      }}
      
      // Continue the detection loop
      if (webcamRunning) {{
        window.requestAnimationFrame(predictWebcam);
      }}
    }}
    
    // Initialize everything
    createPoseLandmarker();
  </script>
</body>
</html>
"""

# Display the component
pose_component = components.html(pose_detector_html, height=600)

# Debug output for easy troubleshooting
if 'debug' not in st.session_state:
    st.session_state.debug = []

# Initialize status display
status_placeholder = st.empty()
if st.session_state.alarm_triggered:
    status_placeholder.warning("⚠️ Previous alert is still active")

# Handle component events
if pose_component and isinstance(pose_component, dict):
    current_time = time.time()
    
    # Add to debug log
    st.session_state.debug.append(f"Component data: {pose_component}")
    
    # Handle hands raised event
    if pose_component.get("hands_raised", False) and enable_alarm and alarm_audio is not None:
        alarm_status.warning("⚠️ ALERT: Hands raised detected for extended period!")
        
        # Play alarm sound if cooldown has passed
        if not st.session_state.alarm_triggered and (current_time - st.session_state.last_alarm_time) > alarm_cooldown:
            st.session_state.last_alarm_time = current_time
            st.session_state.alarm_triggered = True
            audio_b64 = base64.b64encode(alarm_audio).decode()
            alarm_player.markdown(
                f'<audio autoplay><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3"></audio>',
                unsafe_allow_html=True
            )
    elif not pose_component.get("hands_raised", True) and st.session_state.alarm_triggered:
        alarm_status.empty()
        alarm_player.empty()
        st.session_state.alarm_triggered = False

# Show the last 3 debug messages in an expandable section
with st.expander("Debug Information", expanded=False):
    for msg in st.session_state.debug[-3:]:
        st.text(msg)
    if st.button("Clear Debug Log"):
        st.session_state.debug = []

# About section
st.markdown("""
---
## About
This demo uses MediaPipe Pose Detection running in your browser for maximum performance.
The pose detection runs at 25-30 FPS on most modern devices.
""")
