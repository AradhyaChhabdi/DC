import os
import cv2
from flask import Flask, render_template, request, redirect, url_for, session, Response, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO

# --- Application Setup ---
app = Flask(__name__)
app.secret_key = 'a-very-secret-and-secure-key' 

# --- Configuration ---
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Model Loading ---
model = YOLO('yolov8n.pt')

# --- Global State Management ---
SELECTED_TRACK_ID = None
LATEST_RESULTS = None

# --- Core Video Processing Function ---
def generate_frames(video_path):
    """
    Generator function to process an uploaded video file frame by frame.
    It performs object tracking and yields frames for streaming.
    Includes debugging print statements.
    """
    global SELECTED_TRACK_ID, LATEST_RESULTS

    print(f"--- Attempting to open video: {video_path} ---")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"!!! CRITICAL ERROR: Could not open video file at {video_path} !!!")
        return

    print("--- Video opened successfully. Starting frame processing loop. ---")
    frame_count = 0
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("--- End of video or failed to read frame. Exiting loop. ---")
                break

            frame_count += 1
            print(f"Processing frame #{frame_count}")

            # Run YOLOv8 tracking on the frame
            results = model.track(frame, persist=True)
            
            # Store the latest results for the click handler
            if results and len(results) > 0:
                LATEST_RESULTS = results[0]
                annotated_frame = LATEST_RESULTS.plot()
            else:
                # If no results, use the original frame
                annotated_frame = frame

            # If a specific object is being tracked, customize the annotation
            if SELECTED_TRACK_ID is not None and LATEST_RESULTS and LATEST_RESULTS.boxes.id is not None:
                annotated_frame = frame.copy()
                boxes = LATEST_RESULTS.boxes.xyxy.cpu().numpy().astype(int)
                track_ids = LATEST_RESULTS.boxes.id.cpu().numpy().astype(int)

                for box, track_id in zip(boxes, track_ids):
                    if track_id == SELECTED_TRACK_ID:
                        x1, y1, x2, y2 = box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(annotated_frame, f"TRACKING ID: {track_id}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        break

            # Encode the frame to JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                print(f"Warning: Failed to encode frame #{frame_count}")
                continue

            # Yield the frame for streaming
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    except Exception as e:
        print(f"!!! An exception occurred during frame generation: {e} !!!")
    finally:
        # Ensure the video capture is released
        print("--- Releasing video capture object. ---")
        cap.release()
        SELECTED_TRACK_ID = None
        LATEST_RESULTS = None


# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return redirect(request.url)
    
    file = request.files['video']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        session['video_path'] = filepath
        return redirect(url_for('processing'))
    return redirect(request.url)

@app.route('/processing')
def processing():
    video_path = session.get('video_path')
    if not video_path or not os.path.exists(video_path):
        return "Error: Video not found. Please upload again.", 404
    
    return render_template('processing.html')

@app.route('/video_feed')
def video_feed():
    video_path = session.get('video_path', None)
    if not video_path:
        return "Error: No video path in session. Please upload a video.", 400
        
    return Response(generate_frames(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/select_object', methods=['POST'])
def select_object():
    global SELECTED_TRACK_ID, LATEST_RESULTS
    
    if SELECTED_TRACK_ID is not None:
        return jsonify(success=False, message="An object is already tracked. Please reset first.")

    data = request.get_json()
    x, y = int(data['x']), int(data['y'])

    if LATEST_RESULTS and LATEST_RESULTS.boxes.id is not None:
        boxes = LATEST_RESULTS.boxes.xyxy.cpu().numpy().astype(int)
        ids = LATEST_RESULTS.boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = box
            if x1 < x < x2 and y1 < y < y2:
                SELECTED_TRACK_ID = track_id
                return jsonify(success=True, message=f"Locked onto object with ID: {track_id}")

    return jsonify(success=False, message="No object found at the clicked coordinates.")

@app.route('/reset_selection', methods=['POST'])
def reset_selection():
    global SELECTED_TRACK_ID
    SELECTED_TRACK_ID = None
    return jsonify(success=True, message="Selection reset. Detecting all objects.")

# --- Main Execution ---
if __name__ == '__main__':
    app.run(debug=True)
