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

# In-memory store for bounding box and selection data
detection_data = {}

def is_point_in_box(point, box):
    """Check if a point (x, y) is inside a bounding box [x1, y1, x2, y2]."""
    x, y = point
    x1, y1, x2, y2 = box
    return x1 < x < x2 and y1 < y < y2

def generate_frames(video_path):
    """Generator function to process video, handle object detection, and yield frames."""
    session_key = video_path
    detection_data[session_key] = {'boxes': [], 'selected_id': None}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        if session_key in detection_data:
            del detection_data[session_key]
        return

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # --- FIX 1: Access the first element of the results list ---
        results = model(frame)[0]
        
        selected_obj_id = detection_data.get(session_key, {}).get('selected_id')

        boxes = results.boxes.xyxy.tolist()
        classes = results.boxes.cls.tolist()
        confidences = results.boxes.conf.tolist()
        names = results.names
        
        # --- FIX 2: Initialize current_boxes as an empty list ---
        current_boxes = []
        for i, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
            if conf > 0.5:
                current_boxes.append({
                    'id': i,
                    'box': [int(coord) for coord in box],
                    'class': names[int(cls)],
                    'confidence': conf
                })
        
        # --- FIX 3: Use the correct variable 'detection_data' ---
        if session_key in detection_data:
            detection_data[session_key]['boxes'] = current_boxes

        if selected_obj_id is not None:
            for obj in current_boxes:
                if obj['id'] == selected_obj_id:
                    x1, y1, x2, y2 = obj['box']
                    label = f"{obj['class']} {obj['confidence']:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    break
        else:
            # --- FIX 1 (cont.): Use results[0] for plotting ---
            frame = results.plot()

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    if session_key in detection_data:
        del detection_data[session_key]

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
    video_path = session.get('video_path')
    if not video_path:
        return "No video path found in session.", 404
    return Response(generate_frames(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/select_object', methods=['POST'])
def select_object():
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'Invalid JSON'}), 400

    x, y = data['x'], data['y']
    video_path = session.get('video_path')
    session_key = video_path

    if not video_path or session_key not in detection_data:
        return jsonify({'status': 'error', 'message': 'No video or boxes found'}), 400

    boxes_in_frame = detection_data[session_key]['boxes']
    
    selected_box_id = None
    for item in boxes_in_frame:
        if is_point_in_box((x, y), item['box']):
            selected_box_id = item['id']
            break

    if selected_box_id is not None:
        detection_data[session_key]['selected_id'] = selected_box_id
        return jsonify({'status': 'success', 'selected_id': selected_box_id})
    else:
        if session_key in detection_data:
            detection_data[session_key]['selected_id'] = None
        return jsonify({'status': 'miss', 'message': 'No object at this location'})
    
if __name__ == '__main__':
    app.run(debug=True)