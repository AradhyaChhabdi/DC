import os
from flask import Flask, render_template, Response, jsonify, request
import cv2
from ultralytics import YOLO
import airsim
import numpy as np
import threading
import time
import math

app = Flask(__name__)
app.secret_key = 'your_very_secret_key'

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# --- Global variables for tracking state ---
SELECTED_TRACK_ID = None
LATEST_RESULTS = None
CONTROL_MODE = "MANUAL"  # MANUAL or AUTO
TARGET_POSITION_3D = None
TARGET_CENTER_PX = None  # (x, y) center of target in image space
TARGET_DEPTH_M = None
COLLISION_DETECTED = False
AUTO_START_TIME = 0.0  # when AUTO mode was engaged
AUTO_ARM_SECONDS = 1.5
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_LAST_UPDATE = None  # timestamp of last target pixel/depth update

# AirSim client (global to share between threads)
airsim_client = None
airsim_lock = threading.Lock()

# PID Controller parameters
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0
        
    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output
    
    def reset(self):
        self.prev_error = 0
        self.integral = 0

# Initialize PID controllers for steering and throttle
steering_pid = PIDController(kp=0.5, ki=0.0, kd=0.1)
throttle_pid = PIDController(kp=0.3, ki=0.0, kd=0.05)

def get_3d_position_from_2d(client, pixel_x, pixel_y, depth_value):
    """Convert 2D pixel coordinates to 3D world coordinates using depth."""
    # Camera intrinsics (adjust based on your AirSim camera settings)
    # Default AirSim camera FOV is 90 degrees
    image_width = 640
    image_height = 480
    fov = 90  # degrees
    
    # Calculate focal length
    focal_length = (image_width / 2) / math.tan(math.radians(fov / 2))
    
    # Convert pixel to normalized coordinates
    x_normalized = (pixel_x - image_width / 2) / focal_length
    y_normalized = (pixel_y - image_height / 2) / focal_length
    
    # Get car pose
    car_pose = client.simGetVehiclePose()
    car_position = car_pose.position
    car_orientation = car_pose.orientation
    
    # Calculate 3D position in camera frame
    z_cam = depth_value
    x_cam = x_normalized * z_cam
    y_cam = y_normalized * z_cam
    
    # Transform to world coordinates (simplified - assumes camera aligned with car)
    # In reality, you'd need proper transformation matrices
    target_x = car_position.x_val + z_cam * math.cos(car_orientation.z_val) - x_cam * math.sin(car_orientation.z_val)
    target_y = car_position.y_val + z_cam * math.sin(car_orientation.z_val) + x_cam * math.cos(car_orientation.z_val)
    target_z = car_position.z_val + y_cam
    
    return airsim.Vector3r(target_x, target_y, target_z)

def autonomous_navigation_thread():
    """Background thread for autonomous car control."""
    global CONTROL_MODE, TARGET_POSITION_3D, TARGET_CENTER_PX, TARGET_DEPTH_M, COLLISION_DETECTED, AUTO_START_TIME, airsim_client, TARGET_LAST_UPDATE
    
    print("Autonomous navigation thread started")
    
    while True:
        try:
            if CONTROL_MODE == "AUTO" and airsim_client is not None:
                # Get current car state
                with airsim_lock:
                    car_state = airsim_client.getCarState()
                    car_pose = airsim_client.simGetVehiclePose()
                car_pos = car_pose.position
                car_orientation = car_pose.orientation
                
                # Check for collision with arming logic to avoid false positives
                with airsim_lock:
                    collision_info = airsim_client.simGetCollisionInfo()
                # Arm collision checks only after some time and movement
                armed = False
                try:
                    armed = (time.time() - AUTO_START_TIME) > AUTO_ARM_SECONDS and car_state.speed > 0.5
                except Exception:
                    armed = False
                if collision_info.has_collided and armed:
                    print("COLLISION DETECTED (armed)! Switching to MANUAL mode")
                    COLLISION_DETECTED = True
                    CONTROL_MODE = "MANUAL"

                    car_controls = airsim.CarControls()
                    car_controls.throttle = 0
                    car_controls.brake = 1
                    with airsim_lock:
                        airsim_client.setCarControls(car_controls)

                    time.sleep(1)
                    continue
                
                # Prefer image-based visual servoing for robustness
                dt = 0.1  # 10 Hz update rate
                if TARGET_CENTER_PX is not None and FRAME_WIDTH > 0:
                    cx, cy = TARGET_CENTER_PX
                    img_center_x = FRAME_WIDTH / 2.0
                    # Normalize horizontal error to [-1, 1]
                    err_x = (cx - img_center_x) / max(1.0, img_center_x)

                    # Use PID on image-space error for steering
                    steering = steering_pid.compute(err_x, dt)
                    steering = max(-1.0, min(1.0, steering))

                    # Throttle strategy: moderate speed, reduce when turning hard
                    base_throttle = 0.55
                    throttle = base_throttle * (1.0 - min(1.0, abs(steering)))
                    throttle = max(0.3, min(0.65, throttle))

                    # Optional: if depth is available and very close, ease off
                    if TARGET_DEPTH_M is not None and TARGET_DEPTH_M < 1.5:
                        throttle = min(throttle, 0.2)

                    # If target info is stale, slow/stop to avoid blind driving
                    now_ts = time.time()
                    if TARGET_LAST_UPDATE is not None and (now_ts - TARGET_LAST_UPDATE) > 0.6:
                        throttle = min(throttle, 0.15)

                    # For logging
                    distance = None
                    angle_error = err_x
                elif TARGET_POSITION_3D is not None:
                    # Fallback to world-space navigation if available
                    dx = TARGET_POSITION_3D.x_val - car_pos.x_val
                    dy = TARGET_POSITION_3D.y_val - car_pos.y_val
                    distance = math.sqrt(dx**2 + dy**2)

                    target_angle = math.atan2(dy, dx)
                    _, _, car_yaw = airsim.to_eularian_angles(car_orientation)
                    angle_error = target_angle - car_yaw
                    angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))

                    steering = steering_pid.compute(angle_error, dt)
                    steering = max(-1.0, min(1.0, steering))

                    desired_speed = min(distance * 2, 5.0)
                    current_speed = car_state.speed
                    speed_error = desired_speed - current_speed
                    throttle = throttle_pid.compute(speed_error, dt)
                    throttle = max(0.0, min(1.0, throttle))
                else:
                    # No target info; hold still
                    steering = 0.0
                    throttle = 0.0
                    distance = None
                    angle_error = 0.0
                
                # Apply controls
                car_controls = airsim.CarControls()
                car_controls.throttle = throttle
                car_controls.steering = steering
                car_controls.brake = 0
                with airsim_lock:
                    airsim_client.setCarControls(car_controls)
                
                try:
                    if distance is not None:
                        print(f"AUTO Mode - Dist: {distance:.2f}m, Err: {angle_error:.2f}, Steer: {steering:.2f}, Thr: {throttle:.2f}")
                    else:
                        print(f"AUTO Mode (image) - ErrX: {angle_error:.2f}, Steer: {steering:.2f}, Thr: {throttle:.2f}")
                except Exception:
                    pass
                
                # Stop conditions: close in world or depth proximity (image-based)
                stop_due_to_proximity = False
                try:
                    if distance is not None and distance < 2.0:
                        stop_due_to_proximity = True
                    if TARGET_DEPTH_M is not None and TARGET_DEPTH_M < 0.8:
                        stop_due_to_proximity = True
                except Exception:
                    pass

                if stop_due_to_proximity:
                    print("Proximity reached! Stopping...")
                    car_controls = airsim.CarControls()
                    car_controls.throttle = 0
                    car_controls.brake = 1
                    with airsim_lock:
                        airsim_client.setCarControls(car_controls)
                
            time.sleep(0.1)  # 10 Hz update rate
            
        except Exception as e:
            print(f"Error in autonomous navigation: {e}")
            time.sleep(1)

def generate_frames():
    """Generates video frames from an AirSim simulation."""
    global SELECTED_TRACK_ID, LATEST_RESULTS, TARGET_POSITION_3D, TARGET_CENTER_PX, TARGET_DEPTH_M, airsim_client, CONTROL_MODE

    # --- AirSim Connection for Car ---
    airsim_client = airsim.CarClient()
    with airsim_lock:
        airsim_client.confirmConnection()
        airsim_client.enableApiControl(True)
    print("Connected to AirSim!")

    while True:
        try:
            # --- Get Image and Depth from AirSim ---
            with airsim_lock:
                responses = airsim_client.simGetImages([
                    airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
                    airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)
                ])
            
            # RGB image
            response_rgb = responses[0]
            img1d = np.frombuffer(response_rgb.image_data_uint8, dtype=np.uint8)
            frame = img1d.reshape(response_rgb.height, response_rgb.width, 3)
            
            # Depth image
            response_depth = responses[1]
            depth_img = airsim.list_to_2d_float_array(response_depth.image_data_float, 
                                                     response_depth.width, 
                                                     response_depth.height)
            depth_img = np.array(depth_img)

            # --- YOLO Logic ---
            # Force processing at a consistent size for reliable coordinates
            results = model.track(frame, persist=True, imgsz=640)
            LATEST_RESULTS = results[0]
            
            # The annotated frame from plot() is already at the correct size (e.g., 640x480)
            annotated_frame = results[0].plot()

            # Update global frame dimensions from the annotated frame, which is what's sent to the user
            global FRAME_WIDTH, FRAME_HEIGHT
            FRAME_HEIGHT, FRAME_WIDTH, _ = annotated_frame.shape

            if SELECTED_TRACK_ID is not None:
                annotated_frame = frame.copy()
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    ids = results[0].boxes.id.cpu().numpy().astype(int)
                    
                    for box, track_id in zip(boxes, ids):
                        if track_id == SELECTED_TRACK_ID:
                            (x1, y1, x2, y2) = box
                            
                            # Calculate center of bounding box
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            
                            # Get depth at center
                            if 0 <= center_y < depth_img.shape[0] and 0 <= center_x < depth_img.shape[1]:
                                depth_value = depth_img[center_y, center_x]
                                # Store latest image-space target
                                TARGET_CENTER_PX = (center_x, center_y)
                                TARGET_DEPTH_M = float(depth_value)
                                # Mark last update time for target data
                                global TARGET_LAST_UPDATE
                                TARGET_LAST_UPDATE = time.time()

                                # Update 3D target position if in AUTO mode
                                if CONTROL_MODE == "AUTO":
                                    TARGET_POSITION_3D = get_3d_position_from_2d(
                                        airsim_client, center_x, center_y, depth_value
                                    )
                                
                                # Draw tracking box
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                cv2.putText(annotated_frame, f"TRACKING ID: {track_id} | Depth: {depth_value:.2f}m", 
                                          (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 255, 0), -1)
                            break
            
            # Display mode on frame
            mode_color = (0, 255, 0) if CONTROL_MODE == "MANUAL" else (0, 165, 255)
            cv2.putText(annotated_frame, f"MODE: {CONTROL_MODE}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, mode_color, 2)

            # --- Encode and Stream the Frame ---
            bgr_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            (flag, encodedImage) = cv2.imencode(".jpg", bgr_frame)
            if not flag:
                continue
            
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                  bytearray(encodedImage) + b'\r\n')

        except Exception as e:
            print(f"Error connecting to AirSim or processing frame: {e}")
            break

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/select_object', methods=['POST'])
def select_object():
    """Receives click coordinates and locks onto an object."""
    global SELECTED_TRACK_ID, LATEST_RESULTS, CONTROL_MODE, AUTO_START_TIME, COLLISION_DETECTED
    
    if SELECTED_TRACK_ID is not None:
        return jsonify(success=False, message="An object is already tracked. Reset first.")

    data = request.get_json()
    x_click, y_click = int(data['x']), int(data['y'])
    
    print(f"Click received at: ({x_click}, {y_click})")

    if LATEST_RESULTS is None:
        print("No YOLO results available yet")
        return jsonify(success=False, message="No detection results available. Wait a moment.")
    
    if LATEST_RESULTS.boxes.id is None:
        print("No tracked objects in frame")
        return jsonify(success=False, message="No tracked objects detected in frame.")
    
    # Get bounding boxes (Ultralytics returns boxes rescaled to the original frame)
    boxes = LATEST_RESULTS.boxes.xyxy.cpu().numpy().astype(int)
    ids = LATEST_RESULTS.boxes.id.cpu().numpy().astype(int)
    
    # The click coordinates are relative to the *displayed* frame size in the browser.
    # The bounding boxes are relative to the *annotated* frame size from YOLO.
    # We have already set FRAME_WIDTH and FRAME_HEIGHT to the annotated frame's dimensions.
    # The frontend will scale its click based on its display size vs. the frame size we give it.
    # Therefore, the incoming x_click and y_click should be directly comparable to the box coordinates.
    
    x, y = x_click, y_click

    print(f"\n{'='*60}")
    print(f"Comparing click ({x}, {y}) directly against box coordinates.")
    print(f"Frame dimensions sent to frontend: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"Number of boxes: {len(boxes)}")
    try:
        ann_h, ann_w, _ = LATEST_RESULTS.plot().shape
        print(f"Annotated frame shape (server): {ann_w}x{ann_h}")
    except Exception:
        pass
    print(f"{'='*60}")
    
    # First pass: strict inside-box hit test
    for box, track_id in zip(boxes, ids):
        x1, y1, x2, y2 = box
        print(f"Object ID {track_id}: Box[{x1}, {y1}, {x2}, {y2}]")
        print(f"  X check: {x1} <= {x} <= {x2} ? {x1 <= x <= x2}")
        print(f"  Y check: {y1} <= {y} <= {y2} ? {y1 <= y <= y2}")
        
        # Use <= instead of < to include edges
        if x1 <= x <= x2 and y1 <= y <= y2:
            SELECTED_TRACK_ID = track_id
            CONTROL_MODE = "AUTO"
            steering_pid.reset()
            throttle_pid.reset()
            print(f"\n✓✓✓ OBJECT SELECTED! ID: {track_id}, Switching to AUTO mode ✓✓✓\n")
            # Arm AUTO mode
            from time import time as _t
            AUTO_START_TIME = _t()
            COLLISION_DETECTED = False
            return jsonify(success=True, message=f"🎯 TRACKING OBJECT ID: {track_id} | AUTO MODE ACTIVE!", track_id=track_id)

    # Second pass: allow a margin around boxes and nearest-box snapping
    # This helps in case of tiny scale or offset mismatches on the client.
    margin = max(10, int(0.05 * min(FRAME_WIDTH, FRAME_HEIGHT)))
    print(f"No direct hit. Trying margin snap with margin={margin}...")

    best_id = None
    best_dist = 1e9
    for box, track_id in zip(boxes, ids):
        x1, y1, x2, y2 = box
        # Expanded box
        ex1 = max(0, x1 - margin)
        ey1 = max(0, y1 - margin)
        ex2 = min(FRAME_WIDTH - 1, x2 + margin)
        ey2 = min(FRAME_HEIGHT - 1, y2 + margin)

        inside_expanded = (ex1 <= x <= ex2) and (ey1 <= y <= ey2)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
        print(f"  ID {track_id}: expanded [{ex1}, {ey1}, {ex2}, {ey2}], inside_expanded={inside_expanded}, center_dist={dist:.1f}")

        if inside_expanded and dist < best_dist:
            best_dist = dist
            best_id = track_id

    # Accept nearest if reasonably close
    proximity_threshold = max(30, int(0.08 * min(FRAME_WIDTH, FRAME_HEIGHT)))
    if best_id is not None and best_dist <= proximity_threshold:
        SELECTED_TRACK_ID = best_id
        CONTROL_MODE = "AUTO"
        steering_pid.reset()
        throttle_pid.reset()
        print(f"\n✓✓✓ NEAREST OBJECT SELECTED! ID: {best_id}, dist={best_dist:.1f}, Switching to AUTO mode ✓✓✓\n")
        return jsonify(success=True, message=f"🎯 TRACKING NEAREST OBJECT ID: {best_id} | AUTO MODE ACTIVE!", track_id=best_id)

    print(f"\n✗ Click ({x}, {y}) did not hit any bounding box (after margin)")
    print(f"{'='*60}\n")
    return jsonify(success=False, message="No object at that location. Try clicking directly on a detected object.")

@app.route('/reset_selection', methods=['POST'])
def reset_selection():
    """Resets the selected track ID and returns to MANUAL mode."""
    global SELECTED_TRACK_ID, CONTROL_MODE, TARGET_POSITION_3D, TARGET_CENTER_PX, TARGET_DEPTH_M, COLLISION_DETECTED
    
    SELECTED_TRACK_ID = None
    TARGET_POSITION_3D = None
    TARGET_CENTER_PX = None
    TARGET_DEPTH_M = None
    COLLISION_DETECTED = False
    CONTROL_MODE = "MANUAL"
    
    # Stop the car
    if airsim_client is not None:
        car_controls = airsim.CarControls()
        car_controls.throttle = 0
        car_controls.brake = 1
        with airsim_lock:
            airsim_client.setCarControls(car_controls)
    
    global AUTO_START_TIME
    AUTO_START_TIME = 0.0
    return jsonify(success=True, message="Selection reset. MANUAL mode activated.")

@app.route('/select_track', methods=['POST'])
def select_track():
    """Select an object directly by track_id (avoids coordinate mismatches)."""
    global SELECTED_TRACK_ID, LATEST_RESULTS, CONTROL_MODE, AUTO_START_TIME

    if LATEST_RESULTS is None or LATEST_RESULTS.boxes.id is None:
        return jsonify(success=False, message="No tracked objects available.")

    data = request.get_json()
    track_id_req = data.get('track_id')
    if track_id_req is None:
        return jsonify(success=False, message="Missing track_id.")

    ids = LATEST_RESULTS.boxes.id.cpu().numpy().astype(int).tolist()
    print(f"/select_track requested id={track_id_req}; available={ids}")
    if int(track_id_req) not in ids:
        return jsonify(success=False, message=f"Track ID {track_id_req} not in current frame.")

    SELECTED_TRACK_ID = int(track_id_req)
    CONTROL_MODE = "AUTO"
    steering_pid.reset()
    throttle_pid.reset()
    from time import time as _t
    AUTO_START_TIME = _t()
    global COLLISION_DETECTED
    COLLISION_DETECTED = False
    print(f"\n✓✓✓ OBJECT SELECTED VIA ID! ID: {SELECTED_TRACK_ID}, Switching to AUTO mode ✓✓✓\n")
    return jsonify(success=True, message=f"🎯 TRACKING OBJECT ID: {SELECTED_TRACK_ID} | AUTO MODE ACTIVE!", track_id=SELECTED_TRACK_ID)

@app.route('/toggle_control_source', methods=['POST'])
def toggle_control_source():
    """Toggle between API control (browser) and simulation control."""
    global airsim_client
    
    if airsim_client is None:
        return jsonify(success=False, message="AirSim not connected.")
    
    data = request.get_json()
    source = data.get('source')  # 'browser' or 'simulation'
    
    if source == 'simulation':
        # Disable API control to allow simulation keyboard control
        with airsim_lock:
            airsim_client.enableApiControl(False)
        return jsonify(success=True, message="Simulation control enabled. Use AirSim keyboard.")
    elif source == 'browser':
        # Enable API control for browser control
        with airsim_lock:
            airsim_client.enableApiControl(True)
        return jsonify(success=True, message="Browser control enabled. Use arrow keys in browser.")
    else:
        return jsonify(success=False, message="Invalid control source.")

@app.route('/manual_control', methods=['POST'])
def manual_control():
    """Handles manual keyboard control of the car."""
    global CONTROL_MODE, airsim_client
    
    if CONTROL_MODE != "MANUAL":
        return jsonify(success=False, message="Cannot control manually in AUTO mode.")
    
    data = request.get_json()
    action = data.get('action')
    
    if airsim_client is None:
        return jsonify(success=False, message="AirSim not connected.")
    
    car_controls = airsim.CarControls()
    
    # Map keyboard actions to car controls
    if action == 'forward':
        car_controls.throttle = 0.7
        car_controls.brake = 0
    elif action == 'backward':
        car_controls.throttle = -0.5
        car_controls.brake = 0
    elif action == 'left':
        car_controls.throttle = 0.5
        car_controls.steering = -0.7
        car_controls.brake = 0
    elif action == 'right':
        car_controls.throttle = 0.5
        car_controls.steering = 0.7
        car_controls.brake = 0
    elif action == 'brake':
        car_controls.throttle = 0
        car_controls.brake = 1
    elif action == 'stop':
        car_controls.throttle = 0
        car_controls.brake = 0
        car_controls.steering = 0
    else:
        return jsonify(success=False, message="Unknown action.")
    
    with airsim_lock:
        airsim_client.setCarControls(car_controls)
    return jsonify(success=True, message=f"Manual control: {action}")

@app.route('/get_mode', methods=['GET'])
def get_mode():
    """Returns the current control mode."""
    global CONTROL_MODE, COLLISION_DETECTED, AUTO_START_TIME, AUTO_ARM_SECONDS
    now = time.time()
    # Only time-based arming reported here (speed check is in control loop)
    secs_since_auto = max(0.0, now - AUTO_START_TIME)
    arm_secs_remaining = max(0.0, AUTO_ARM_SECONDS - secs_since_auto) if CONTROL_MODE == 'AUTO' else 0.0
    auto_arming = CONTROL_MODE == 'AUTO' and arm_secs_remaining > 0.0
    return jsonify(
        mode=CONTROL_MODE,
        collision_detected=COLLISION_DETECTED,
        auto_arming=auto_arming,
        arm_secs_remaining=round(arm_secs_remaining, 2)
    )

@app.route('/get_frame_size', methods=['GET'])
def get_frame_size():
    """Returns the actual frame dimensions."""
    global FRAME_WIDTH, FRAME_HEIGHT
    return jsonify(width=FRAME_WIDTH, height=FRAME_HEIGHT)

@app.route('/get_current_detections', methods=['GET'])
def get_current_detections():
    """Returns current detection boxes for debugging."""
    global LATEST_RESULTS
    if LATEST_RESULTS is None or LATEST_RESULTS.boxes.id is None:
        return jsonify(detections=[])
    
    boxes = LATEST_RESULTS.boxes.xyxy.cpu().numpy().astype(int).tolist()
    ids = LATEST_RESULTS.boxes.id.cpu().numpy().astype(int).tolist()
    
    detections = []
    for box, track_id in zip(boxes, ids):
        detections.append({
            'id': int(track_id),
            'box': box
        })
    
    return jsonify(detections=detections)

if __name__ == '__main__':
    # Start autonomous navigation thread
    nav_thread = threading.Thread(target=autonomous_navigation_thread, daemon=True)
    nav_thread.start()
    
    app.run(debug=True, use_reloader=False)  # use_reloader=False to prevent double thread creation