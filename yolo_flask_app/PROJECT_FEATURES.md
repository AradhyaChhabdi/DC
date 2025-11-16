# 🚗 Autonomous Vehicle Navigation System - Feature Documentation

## Overview
This project implements a complete autonomous navigation system using computer vision, object tracking, and PID control algorithms in an AirSim simulation environment.

---

## ✅ Implemented Features

### 1. **Multi-Object Tracking**
- **Algorithm**: YOLOv8's built-in ByteTrack algorithm
- **Functionality**: Tracks multiple objects with persistent IDs across frames
- **Usage**: `model.track(frame, persist=True)`

### 2. **Depth Perception & 3D Coordinate Estimation**
- **Depth Camera**: Uses AirSim's DepthPerspective camera
- **3D Position Calculation**: Converts 2D pixel coordinates + depth to 3D world coordinates
- **Function**: `get_3d_position_from_2d()`
- **Output**: Real-world (x, y, z) coordinates of the target object

### 3. **PID Control System**
- **Two PID Controllers**:
  - **Steering PID**: Controls angular direction to target
  - **Throttle PID**: Controls speed based on distance to target
- **Parameters**:
  ```python
  steering_pid = PIDController(kp=0.5, ki=0.0, kd=0.1)
  throttle_pid = PIDController(kp=0.3, ki=0.0, kd=0.05)
  ```

### 4. **Dual Control Modes**

#### **MANUAL Mode** 
- User controls the car with keyboard/buttons
- **Controls**:
  - ↑ / W: Forward
  - ↓ / S: Backward  
  - ← / A: Turn Left
  - → / D: Turn Right
  - Space: Brake

#### **AUTO Mode**
- Automatically activated when user selects an object
- Car autonomously navigates to the target
- Uses PID control for steering and throttle
- Tracks moving objects in real-time

### 5. **Object Selection & Tracking**
- Click on any detected object in the video feed
- System locks onto the object with a unique tracking ID
- Red bounding box highlights the selected target
- Displays depth information in meters

### 6. **Collision Detection**
- Monitors collision events using `simGetCollisionInfo()`
- Automatically switches to MANUAL mode on collision
- Applies emergency brake
- Displays collision alert in UI

### 7. **State Management**
- **States**: MANUAL ↔ AUTO
- **Transitions**:
  - MANUAL → AUTO: When object is selected
  - AUTO → MANUAL: After collision or manual reset
- **Reset Function**: Returns control to user, clears target

### 8. **Real-Time Video Streaming**
- Streams processed video with bounding boxes
- Displays current mode (MANUAL/AUTO) on video
- Shows depth information for tracked objects
- Frame-by-frame YOLO inference

### 9. **Multi-Threading Architecture**
- **Main Thread**: Flask server, video streaming
- **Navigation Thread**: Background autonomous control loop
- **Update Rate**: 10 Hz (0.1s intervals)

---

## 🎯 Complete Workflow

### Step 1: Manual Driving
```
User launches app → MANUAL mode active → User drives with keyboard
```

### Step 2: Object Selection
```
User clicks on object → Object locked with ID → AUTO mode activated
```

### Step 3: Autonomous Navigation
```
System calculates 3D position → PID control engages → Car navigates to target
```

### Step 4: Collision & Reset
```
Car hits target → Collision detected → AUTO mode stops → User clicks Reset → Back to MANUAL
```

---

## 🧮 Technical Components

### Computer Vision Pipeline
1. **Image Acquisition**: RGB + Depth from AirSim cameras
2. **Object Detection**: YOLOv8 inference on RGB frames
3. **Multi-Object Tracking**: ByteTrack maintains IDs across frames
4. **Depth Mapping**: Extract depth value at object center
5. **3D Localization**: Convert 2D + depth → 3D world coordinates

### Navigation Algorithm
1. **Target Acquisition**: Get target 3D position from vision system
2. **State Estimation**: Get car position, orientation, and speed
3. **Error Calculation**: 
   - Distance error: `sqrt((target_x - car_x)² + (target_y - car_y)²)`
   - Angle error: `atan2(dy, dx) - car_yaw`
4. **PID Control**:
   - Steering: Correct angular error
   - Throttle: Maintain speed proportional to distance
5. **Actuation**: Send controls to AirSim

### Control Loop (10 Hz)
```python
while AUTO_MODE:
    1. Get car state (position, orientation, speed)
    2. Check for collision
    3. Calculate distance & angle to target
    4. Compute PID outputs
    5. Apply steering & throttle
    6. Sleep 0.1s
```

---

## 📊 Key Algorithms

### ByteTrack (Multi-Object Tracking)
- Built into YOLOv8
- Tracks objects across frames with unique IDs
- Handles occlusions and re-identification

### PID Control Formula
```
output = Kp × error + Ki × ∫error·dt + Kd × d(error)/dt
```

### 3D Coordinate Transformation
```python
# Camera intrinsics → Normalized coordinates → 3D camera frame → World frame
focal_length = (width/2) / tan(FOV/2)
x_cam = (pixel_x - cx) / focal_length × depth
y_cam = (pixel_y - cy) / focal_length × depth
# Transform to world using car pose
```

---

## 🎓 For Professor Presentation

**Key Points to Emphasize:**

1. **Complete CV Pipeline**: Detection → Tracking → Depth → 3D Localization
2. **Control Theory**: PID controllers for real-time vehicle control
3. **State Machine**: Seamless transitions between manual and autonomous modes
4. **Safety**: Collision detection and automatic failsafe to manual mode
5. **Real-Time Performance**: 10 Hz control loop, streaming video processing
6. **Simulation Integration**: Full AirSim API usage for realistic vehicle dynamics

**Demonstrates Knowledge Of:**
- Computer Vision (YOLO, object detection, tracking)
- 3D Geometry (coordinate transformations, camera calibration)
- Control Systems (PID, state feedback)
- Robotics (autonomous navigation, sensor fusion)
- Software Engineering (multi-threading, real-time systems, web interfaces)

---

## 🚀 Running the Project

1. **Start AirSim** (Blocks environment with car)
2. **Activate Python environment**:
   ```powershell
   cd yolo_flask_app
   .\myenv\Scripts\Activate.ps1
   ```
3. **Run Flask app**:
   ```powershell
   python app.py
   ```
4. **Open browser**: `http://127.0.0.1:5000`
5. **Drive manually** with arrow keys
6. **Click on object** to engage autonomous mode
7. **Watch car navigate** to target and collide
8. **Click Reset** to regain manual control

---

## 📝 Future Enhancements (Optional to Mention)

- Path planning algorithms (A*, RRT)
- Obstacle avoidance while pursuing target
- Kalman filter for smoother target tracking
- Deep learning for end-to-end control
- Multiple target selection and prioritization
