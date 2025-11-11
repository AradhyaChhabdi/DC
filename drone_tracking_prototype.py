"""
Drone Tracking Prototype - Pure Python + OpenCV + YOLOv8
Demonstrates: Car detection, selection, tracking, and virtual drone following/landing
"""

import cv2
import numpy as np
from ultralytics import YOLO

# --- Configuration ---
# PID Controller Parameters
KP = 0.08  # Proportional gain (how strongly to move toward target)
KI = 0.001  # Integral gain (corrects steady-state error)
KD = 0.15  # Derivative gain (dampening/smoothing)

# Motion Parameters
MAX_SPEED = 15.0  # Maximum pixels per frame
ACCELERATION = 0.8  # How quickly drone accelerates (0-1)
DECELERATION = 0.9  # Velocity decay factor (0-1, closer to 1 = smoother)

# Distance Thresholds
LOCKED_THRESHOLD = 50  # Pixels - within this range is "LOCKED"
LANDING_THRESHOLD = 25  # Pixels - within this range is "LANDING"
LANDING_SUCCESS_THRESHOLD = 10  # Pixels - successful landing

# Altitude Simulation
INITIAL_ALTITUDE = 10.0  # Starting altitude in meters
ALTITUDE_DESCENT_RATE = 0.02  # Meters per frame when approaching
MIN_ALTITUDE = 0.3  # Minimum altitude before landing
LANDING_SUCCESS_ALTITUDE = 0.2  # Altitude threshold for successful landing

# Visual Settings
DRONE_COLOR = (0, 255, 0)  # Green
LOCKED_COLOR = (0, 255, 255)  # Yellow when locked
LANDING_COLOR = (0, 0, 255)  # Red when landing
SUCCESS_COLOR = (0, 255, 0)  # Green when landed
CROSSHAIR_SIZE = 20

# --- Global State ---
selected_track_id = None
drone_position = None  # (x, y) - virtual drone position
drone_velocity = np.array([0.0, 0.0])  # Velocity vector (vx, vy)
drone_altitude = INITIAL_ALTITUDE  # Current altitude in meters
pid_integral = np.array([0.0, 0.0])  # PID integral term
previous_error = np.array([0.0, 0.0])  # Previous error for derivative
is_landing = False
landing_successful = False
click_coords = None
drone_state = "IDLE"  # States: IDLE, APPROACHING, LOCKED, LANDING, LANDED


def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks to select objects."""
    global click_coords
    if event == cv2.EVENT_LBUTTONDOWN:
        click_coords = (x, y)


def draw_crosshair(frame, center, color, size=CROSSHAIR_SIZE, thickness=2):
    """Draw a crosshair representing the virtual drone."""
    x, y = center
    cv2.line(frame, (x - size, y), (x + size, y), color, thickness)
    cv2.line(frame, (x, y - size), (x, y + size), color, thickness)
    cv2.circle(frame, (x, y), 5, color, -1)
    cv2.circle(frame, (x, y), size, color, thickness)


def draw_landing_animation(frame, center, radius):
    """Draw animated landing circles."""
    for i in range(3):
        r = radius + i * 15
        alpha = 255 - i * 60
        color = (0, int(alpha * 0.5), int(alpha))
        cv2.circle(frame, center, r, color, 2)


def select_object_at_click(boxes, ids, click_pos):
    """Find which tracked object was clicked."""
    x_click, y_click = click_pos
    for box, track_id in zip(boxes, ids):
        x1, y1, x2, y2 = box
        if x1 < x_click < x2 and y1 < y_click < y2:
            return track_id, ((x1 + x2) // 2, (y1 + y2) // 2)
    return None, None


def pid_controller(current_pos, target_pos, velocity, integral, prev_error):
    """
    PID controller for smooth drone movement with realistic physics.
    Returns: new_position, new_velocity, new_integral, current_error
    """
    current_pos = np.array(current_pos, dtype=float)
    target_pos = np.array(target_pos, dtype=float)
    
    # Calculate error (distance to target)
    error = target_pos - current_pos
    
    # PID terms
    proportional = KP * error
    integral = integral + KI * error
    derivative = KD * (error - prev_error)
    
    # Calculate desired acceleration
    acceleration = proportional + integral + derivative
    
    # Update velocity with acceleration and deceleration
    velocity = velocity * DECELERATION + acceleration * ACCELERATION
    
    # Limit maximum speed
    speed = np.linalg.norm(velocity)
    if speed > MAX_SPEED:
        velocity = velocity / speed * MAX_SPEED
    
    # Update position
    new_position = current_pos + velocity
    
    return new_position, velocity, integral, error


def update_altitude(current_altitude, distance_to_target):
    """Update drone altitude based on distance to target."""
    if distance_to_target < LOCKED_THRESHOLD:
        # Descend when close to target
        descent = ALTITUDE_DESCENT_RATE * (1.0 - distance_to_target / LOCKED_THRESHOLD)
        new_altitude = max(MIN_ALTITUDE, current_altitude - descent)
    else:
        # Slowly increase altitude when far from target
        new_altitude = min(INITIAL_ALTITUDE, current_altitude + ALTITUDE_DESCENT_RATE * 0.3)
    
    return new_altitude


def get_drone_state(distance, altitude):
    """Determine drone state based on distance and altitude."""
    if altitude <= LANDING_SUCCESS_ALTITUDE and distance < LANDING_SUCCESS_THRESHOLD:
        return "LANDED"
    elif distance < LANDING_THRESHOLD:
        return "LANDING"
    elif distance < LOCKED_THRESHOLD:
        return "LOCKED"
    else:
        return "APPROACHING"


def draw_velocity_vector(frame, position, velocity, color):
    """Draw an arrow showing velocity direction and magnitude."""
    if np.linalg.norm(velocity) < 0.5:
        return
    
    start_point = tuple(position.astype(int))
    end_point = tuple((position + velocity * 3).astype(int))
    
    cv2.arrowedLine(frame, start_point, end_point, color, 2, tipLength=0.3)


def draw_altitude_bar(frame, altitude, x=30, y=150, width=20, height=150):
    """Draw a vertical bar showing current altitude."""
    # Background bar
    cv2.rectangle(frame, (x, y), (x + width, y + height), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 2)
    
    # Fill based on altitude
    fill_ratio = min(1.0, altitude / INITIAL_ALTITUDE)
    fill_height = int(height * fill_ratio)
    
    if fill_ratio > 0.5:
        bar_color = (0, 255, 0)  # Green
    elif fill_ratio > 0.2:
        bar_color = (0, 255, 255)  # Yellow
    else:
        bar_color = (0, 0, 255)  # Red
    
    if fill_height > 0:
        cv2.rectangle(frame, (x, y + height - fill_height), (x + width, y + height), 
                     bar_color, -1)
    
    # Altitude text
    cv2.putText(frame, f"{altitude:.1f}m", (x - 10, y - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def process_video(video_source=0):
    """
    Main processing loop.
    video_source: 0 for webcam, or path to video file
    """
    global selected_track_id, drone_position, is_landing, click_coords
    global drone_velocity, drone_altitude, pid_integral, previous_error
    global landing_successful, drone_state
    
    # Initialize
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_source}")
        return
    
    cv2.namedWindow('Drone Tracking Prototype')
    cv2.setMouseCallback('Drone Tracking Prototype', mouse_callback)
    
    print("\n" + "="*60)
    print("🚁 DRONE TRACKING PROTOTYPE")
    print("="*60)
    print("📹 Video feed started")
    print("🖱️  Click on a car to select it as target")
    print("🎯 Virtual drone will follow and 'land' on selected car")
    print("⌨️  Press 'r' to reset selection")
    print("⌨️  Press 'q' to quit")
    print("="*60 + "\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop video if it's a file
            if isinstance(video_source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break
        
        frame_count += 1
        h, w = frame.shape[:2]
        
        # Initialize drone position at center if not set
        if drone_position is None:
            drone_position = np.array([w // 2, h // 2], dtype=float)
        
        # --- YOLO Detection and Tracking ---
        results = model.track(frame, persist=True, classes=[2, 5, 7])  # car, bus, truck
        
        # Draw all detections
        annotated_frame = results[0].plot()
        
        # --- Process tracked objects ---
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            labels = results[0].boxes.cls.cpu().numpy().astype(int)
            
            # --- Handle click selection ---
            if click_coords is not None:
                if selected_track_id is None:
                    new_id, target_center = select_object_at_click(boxes, ids, click_coords)
                    if new_id is not None:
                        selected_track_id = new_id
                        print(f"✅ Target locked: ID {selected_track_id}")
                click_coords = None
            
            # --- Track selected object ---
            target_center = None
            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = box
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                
                if track_id == selected_track_id:
                    target_center = center
                    # Highlight selected target
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.putText(annotated_frame, f"TARGET ID: {track_id}", 
                              (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.7, (255, 0, 255), 2)
                    break
            
            # --- Move virtual drone with PID controller ---
            if target_center is not None:
                # Update drone position using PID controller
                drone_position, drone_velocity, pid_integral, previous_error = pid_controller(
                    drone_position, target_center, drone_velocity, pid_integral, previous_error
                )
                
                # Calculate distance to target
                distance = np.linalg.norm(np.array(target_center) - drone_position)
                
                # Update altitude
                drone_altitude = update_altitude(drone_altitude, distance)
                
                # Determine drone state
                drone_state = get_drone_state(distance, drone_altitude)
                
                # Check landing conditions
                is_landing = drone_state in ["LANDING", "LANDED"]
                landing_successful = drone_state == "LANDED"
            else:
                # Target lost - reset to idle
                drone_state = "IDLE"
                landing_successful = False
        
        # --- Draw Virtual Drone ---
        if selected_track_id is not None:
            # Choose color based on state
            if landing_successful:
                color = SUCCESS_COLOR
            elif is_landing:
                color = LANDING_COLOR
            elif drone_state == "LOCKED":
                color = LOCKED_COLOR
            else:
                color = DRONE_COLOR
            
            # Draw drone crosshair
            drone_pos_int = tuple(drone_position.astype(int))
            draw_crosshair(annotated_frame, drone_pos_int, color, size=CROSSHAIR_SIZE)
            
            # Draw velocity vector
            if not landing_successful:
                draw_velocity_vector(annotated_frame, drone_position, drone_velocity, color)
            
            # Draw landing animation
            if is_landing and not landing_successful:
                draw_landing_animation(annotated_frame, drone_pos_int, CROSSHAIR_SIZE + 10)
            
            # Draw state label
            label_offset_y = -40
            if landing_successful:
                state_text = "✓ LANDED!"
                state_color = SUCCESS_COLOR
            else:
                state_text = drone_state
                state_color = color
            
            cv2.putText(annotated_frame, state_text, 
                       (drone_pos_int[0] - 60, drone_pos_int[1] + label_offset_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
            
            # Draw altitude bar
            draw_altitude_bar(annotated_frame, drone_altitude)
        
        # --- Draw UI Info ---
        info_y = 30
        cv2.rectangle(annotated_frame, (10, 10), (450, 150), (0, 0, 0), -1)
        cv2.rectangle(annotated_frame, (10, 10), (450, 150), (255, 255, 255), 2)
        
        cv2.putText(annotated_frame, "Drone Tracking Prototype - Enhanced", 
                   (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += 25
        
        if selected_track_id is None:
            status = "Waiting for target selection..."
            status_color = (200, 200, 200)
        elif landing_successful:
            status = f"Status: LANDED on ID {selected_track_id} ✓"
            status_color = SUCCESS_COLOR
        else:
            status = f"Status: {drone_state} - ID {selected_track_id}"
            status_color = DRONE_COLOR
        
        cv2.putText(annotated_frame, status, (20, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        info_y += 22
        
        # Show altitude and velocity info
        if selected_track_id is not None:
            speed = np.linalg.norm(drone_velocity)
            cv2.putText(annotated_frame, f"Altitude: {drone_altitude:.2f}m | Speed: {speed:.1f}px/f", 
                       (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            info_y += 20
        
        cv2.putText(annotated_frame, "Click car to select | 'r' reset | 'q' quit", 
                   (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        # --- Display ---
        cv2.imshow('Drone Tracking Prototype', annotated_frame)
        
        # --- Keyboard Controls ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n👋 Shutting down...")
            break
        elif key == ord('r'):
            selected_track_id = None
            is_landing = False
            landing_successful = False
            drone_position = np.array([w // 2, h // 2], dtype=float)
            drone_velocity = np.array([0.0, 0.0])
            drone_altitude = INITIAL_ALTITUDE
            pid_integral = np.array([0.0, 0.0])
            previous_error = np.array([0.0, 0.0])
            drone_state = "IDLE"
            print("🔄 Selection reset")
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Prototype closed successfully\n")


if __name__ == '__main__':
    import sys
    
    # Check if video file path is provided
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        print(f"📁 Loading video: {video_path}")
        process_video(video_path)
    else:
        print("📹 Using webcam (default)")
        print("💡 Tip: Run with video file: python drone_tracking_prototype.py <video_path>")
        print()
        process_video(0)  # Use webcam
