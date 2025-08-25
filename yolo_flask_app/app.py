import os
from flask import Flask, render_template, request, redirect, url_for, session, Response, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2

app = Flask(__name__)
app.secret_key = 'your_very_secret_key'  # Replace with a real secret key

# Define the upload folder and ensure it exists
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Global variables for tracking state
SELECTED_TRACK_ID=None

# This will store the latest results from the model for click detection
LATEST_RESULTS=None

# In-memory store for bounding box and selection data
detection_data = {}

def is_point_in_box(point, box):
    """Check if a point (x, y) is inside a bounding box [x1, y1, x2, y2]."""
    x, y = point
    x1, y1, x2, y2 = box
    return x1 < x < x2 and y1 < y < y2

def generate_frames():
    """Generates video frames with object detection and tracking."""
    global SELECTED_TRACK_ID, LATEST_RESULTS
    
    # Use 0 for webcam or provide a path to a video file
    cap = cv2.VideoCapture(0) 

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)
        
        # Store the latest results globally so the click handler can access them
        LATEST_RESULTS = results

        # By default, plot all detections
        annotated_frame = results.plot()

        # If a specific object is being tracked, modify the frame
        if SELECTED_TRACK_ID is not None:
            # Create a clean copy of the frame to draw on
            annotated_frame = frame.copy()
            
            if results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy().astype(int)
                ids = results.boxes.id.cpu().numpy().astype(int)

                for box, track_id in zip(boxes, ids):
                    # If the current object is the one we are tracking
                    if track_id == SELECTED_TRACK_ID:
                        (x1, y1, x2, y2) = box
                        # Draw a more prominent bounding box for the tracked object
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3) # Red, thicker box
                        cv2.putText(annotated_frame, f"TRACKING ID: {track_id}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        break # No need to check other boxes

        # Encode the frame in JPEG format
        (flag, encodedImage) = cv2.imencode(".jpg", annotated_frame)
        if not flag:
            continue

        # Yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encodedImage) + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return redirect(request.url)
    
    file = request.files['video']
    if file and file.filename:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        # --- FIX 4: Correct the redirect flow ---
        session['video_path'] = filepath
        return redirect(url_for('processing'))

    return redirect(request.url)

@app.route('/processing')
def processing():
    video_path = session.get('video_path')
    if not video_path or not os.path.exists(video_path):
        return "Error: Video not found. Please upload again.", 404
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    return render_template('processing.html', video_width=width, video_height=height)

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/select_object', methods=['POST'])
def select_object():
    """
    Receives click coordinates. If no object is currently tracked,
    it finds the object at the coordinates and locks onto it.
    """
    global SELECTED_TRACK_ID, LATEST_RESULTS
    
    # This is the key change: only allow selection if no object is currently tracked.
    if SELECTED_TRACK_ID is not None:
        return jsonify(success=False, message="An object is already being tracked. Please reset first.")

    data = request.get_json()
    x, y = data['x'], data['y']

    if LATEST_RESULTS is not None and LATEST_RESULTS.boxes.id is not None:
        boxes = LATEST_RESULTS.boxes.xyxy.cpu().numpy().astype(int)
        ids = LATEST_RESULTS.boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = box
            # Check if the click coordinates are inside the bounding box
            if x1 < x < x2 and y1 < y < y2:
                SELECTED_TRACK_ID = track_id
                return jsonify(success=True, message=f"Locked onto object with ID: {track_id}")

    return jsonify(success=False, message="No object found at the clicked coordinates.")

@app.route('/reset_selection', methods=['POST'])
def reset_selection():
    """Resets the selected track ID, allowing for a new selection."""
    global SELECTED_TRACK_ID
    SELECTED_TRACK_ID = None
    return jsonify(success=True, message="Selection reset. Ready to track a new object.")