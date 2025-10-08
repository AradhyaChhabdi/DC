import os
from flask import Flask, render_template, Response, jsonify, request
import cv2
from ultralytics import YOLO
import airsim
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_very_secret_key'

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# --- Global variables for tracking state ---
SELECTED_TRACK_ID = None
LATEST_RESULTS = None

def generate_frames():
    """Generates video frames from an AirSim simulation."""
    global SELECTED_TRACK_ID, LATEST_RESULTS

    # --- AirSim Connection for Car ---
    client = airsim.CarClient()
    client.confirmConnection()
    print("Connected to AirSim!")

    while True:
        try:
            # --- Get Image from AirSim ---
            responses = client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            ])
            response = responses[0]

            # --- Convert to OpenCV format ---
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            frame = img1d.reshape(response.height, response.width, 3)

            # --- YOLO Logic ---
            results = model.track(frame, persist=True)
            LATEST_RESULTS = results[0]

            annotated_frame = results[0].plot()

            if SELECTED_TRACK_ID is not None:
                annotated_frame = frame.copy()
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    ids = results[0].boxes.id.cpu().numpy().astype(int)
                    for box, track_id in zip(boxes, ids):
                        if track_id == SELECTED_TRACK_ID:
                            (x1, y1, x2, y2) = box
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv2.putText(annotated_frame, f"TRACKING ID: {track_id}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                            break

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
    global SELECTED_TRACK_ID, LATEST_RESULTS
    
    if SELECTED_TRACK_ID is not None:
        return jsonify(success=False, message="An object is already tracked. Reset first.")

    data = request.get_json()
    x, y = int(data['x']), int(data['y'])

    if LATEST_RESULTS is not None and LATEST_RESULTS.boxes.id is not None:
        boxes = LATEST_RESULTS.boxes.xyxy.cpu().numpy().astype(int)
        ids = LATEST_RESULTS.boxes.id.cpu().numpy().astype(int)
        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = box
            if x1 < x < x2 and y1 < y < y2:
                SELECTED_TRACK_ID = track_id
                return jsonify(success=True, message=f"Locked on ID: {track_id}")

    return jsonify(success=False, message="No object found at coordinates.")

@app.route('/reset_selection', methods=['POST'])
def reset_selection():
    """Resets the selected track ID."""
    global SELECTED_TRACK_ID
    SELECTED_TRACK_ID = None
    return jsonify(success=True, message="Selection reset.")

if __name__ == '__main__':
    app.run(debug=True)