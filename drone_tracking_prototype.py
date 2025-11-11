"""
Drone Tracking Prototype - Pure Python + OpenCV + YOLOv8
Demonstrates: Car detection, selection, tracking, and virtual drone following/landing
"""

import cv2
import numpy as np
from ultralytics import YOLO

# --- Configuration ---
DRONE_SPEED = 0.15  # How fast the virtual drone moves (0-1, lower = smoother)
LANDING_THRESHOLD = 30  # Pixels - when drone is this close, consider "landing"
DRONE_COLOR = (0, 255, 0)  # Green
LANDING_COLOR = (0, 0, 255)  # Red when landing
CROSSHAIR_SIZE = 20

# --- Global State ---
selected_track_id = None
drone_position = None  # (x, y) - virtual drone position
is_landing = False
click_coords = None


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


def move_drone_towards(current_pos, target_pos, speed):
    """Smoothly move drone towards target position."""
    if current_pos is None:
        return target_pos
    
    cx, cy = current_pos
    tx, ty = target_pos
    
    # Calculate direction vector
    dx = tx - cx
    dy = ty - cy
    distance = np.sqrt(dx**2 + dy**2)
    
    if distance < 2:  # Already at target
        return target_pos
    
    # Move towards target with speed factor
    move_distance = distance * speed
    ratio = move_distance / distance
    
    new_x = int(cx + dx * ratio)
    new_y = int(cy + dy * ratio)
    
    return (new_x, new_y)


def process_video(video_source=0):
    """
    Main processing loop.
    video_source: 0 for webcam, or path to video file
    """
    global selected_track_id, drone_position, is_landing, click_coords
    
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
            drone_position = (w // 2, h // 2)
        
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
            
            # --- Move virtual drone ---
            if target_center is not None:
                drone_position = move_drone_towards(drone_position, target_center, DRONE_SPEED)
                
                # Calculate distance to target
                dx = target_center[0] - drone_position[0]
                dy = target_center[1] - drone_position[1]
                distance = np.sqrt(dx**2 + dy**2)
                
                # Check if landing
                if distance < LANDING_THRESHOLD:
                    is_landing = True
                else:
                    is_landing = False
        
        # --- Draw Virtual Drone ---
        if selected_track_id is not None:
            color = LANDING_COLOR if is_landing else DRONE_COLOR
            draw_crosshair(annotated_frame, drone_position, color, size=CROSSHAIR_SIZE)
            
            # Draw landing animation
            if is_landing:
                draw_landing_animation(annotated_frame, drone_position, CROSSHAIR_SIZE + 10)
                cv2.putText(annotated_frame, "LANDING!", 
                          (drone_position[0] - 50, drone_position[1] - 40),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, LANDING_COLOR, 2)
            else:
                cv2.putText(annotated_frame, "FOLLOWING", 
                          (drone_position[0] - 60, drone_position[1] - 40),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, DRONE_COLOR, 2)
        
        # --- Draw UI Info ---
        info_y = 30
        cv2.rectangle(annotated_frame, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.rectangle(annotated_frame, (10, 10), (400, 120), (255, 255, 255), 2)
        
        cv2.putText(annotated_frame, "Drone Tracking Prototype", 
                   (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += 25
        
        if selected_track_id is None:
            status = "Waiting for target selection..."
            status_color = (200, 200, 200)
        elif is_landing:
            status = f"Status: LANDING on ID {selected_track_id}"
            status_color = LANDING_COLOR
        else:
            status = f"Status: FOLLOWING ID {selected_track_id}"
            status_color = DRONE_COLOR
        
        cv2.putText(annotated_frame, status, (20, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
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
            drone_position = (w // 2, h // 2)
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
