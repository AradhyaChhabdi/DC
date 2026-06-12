# 🚗 Autonomous Drone Car Navigation System

A real-time autonomous vehicle navigation system built with **YOLOv8**, **AirSim**, and **Flask**. The car operates inside an Unreal Engine simulation, streams live camera footage to a web browser, and autonomously chases any object the user clicks on — using depth perception, multi-object tracking, and PID control.

---

## 📖 Project Overview

This project bridges **computer vision** and **control systems** to create a fully autonomous vehicle inside a simulation. From the browser, a user can:

- Drive the car manually with keyboard input
- Click on any detected object in the live video feed to hand control over to the autonomous system
- Watch the car track, approach, and follow the selected object in real-time
- Receive collision alerts and seamlessly return to manual control

The system fuses two AirSim camera feeds (RGB + depth), runs YOLOv8 inference on every frame, and uses a 10 Hz PID control loop running in a background thread to steer the vehicle.

---

## 🧱 Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      AirSim (Unreal Engine)              │
│   • RGB camera feed (Scene)                              │
│   • Depth camera feed (DepthPerspective)                 │
│   • Car dynamics & physics                               │
└──────────────┬───────────────────────────────────────────┘
               │  AirSim Python API
               ▼
┌──────────────────────────────────────────────────────────┐
│                      Flask Backend (app.py)              │
│                                                          │
│  ┌─────────────────────┐   ┌──────────────────────────┐ │
│  │   generate_frames() │   │ autonomous_navigation    │ │
│  │   (main thread)     │   │ _thread() (daemon, 10Hz) │ │
│  │                     │   │                          │ │
│  │  • YOLOv8 inference │   │  • PID steering          │ │
│  │  • ByteTrack IDs    │   │  • PID throttle          │ │
│  │  • Depth extraction │   │  • Collision detection   │ │
│  │  • MJPEG streaming  │   │  • Stuck recovery        │ │
│  └─────────────────────┘   └──────────────────────────┘ │
│                                                          │
│  REST API routes: /select_object  /select_track          │
│                   /reset_selection  /manual_control      │
│                   /get_mode  /get_frame_size             │
│                   /get_current_detections                │
└──────────────┬───────────────────────────────────────────┘
               │  HTTP / MJPEG stream
               ▼
┌──────────────────────────────────────────────────────────┐
│                  Browser UI (index.html)                 │
│   • Live MJPEG video with YOLO bounding boxes            │
│   • Canvas overlay (click zones, crosshair, IDs)         │
│   • Keyboard manual control (WASD / arrow keys)          │
│   • Click-to-track object selection                      │
│   • Mode status: MANUAL / AUTO / ARMING / COLLISION      │
└──────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Multi-Object Tracking** | YOLOv8 + ByteTrack assigns persistent IDs across frames |
| **Depth Perception** | AirSim DepthPerspective camera provides per-pixel distance in metres |
| **3D Localisation** | Converts 2D pixel + depth → 3D world coordinates using camera intrinsics |
| **Dual PID Control** | Separate PID controllers for steering (image-space error) and throttle |
| **Visual Servoing** | Primary control loop keeps the target centred in the image frame |
| **Collision Safety** | Armed collision detection → automatic MANUAL fallback + emergency brake |
| **Stuck Recovery** | Detects zero-speed state and applies a throttle burst to break static friction |
| **MJPEG Streaming** | Annotated frames streamed directly to the browser at full speed |
| **Canvas Overlay** | Transparent click-zone highlights and live crosshair drawn over the video |

---

## 📁 Project Structure

```
DC/
├── requirements.txt                   # Python dependencies
├── yolov8n.pt                         # Pre-trained YOLOv8 nano weights
└── yolo_flask_app/
    ├── app.py                         # Flask server + all backend logic
    ├── yolov8n.pt                     # Model weights (mirror in app dir)
    ├── PROJECT_FEATURES.md            # Detailed feature reference
    ├── templates/
    │   ├── index.html                 # Main UI (video feed, controls, overlay)
    │   └── processing.html            # Alternate single-page tracking view
    └── static/
        ├── js/
        │   └── main.js                # Shared JS utilities
        └── uploads/                   # Runtime upload directory
```

---

## ⚙️ How It Works

### 1. Video Pipeline
Every iteration of `generate_frames()`:
1. Captures an **RGB** frame and a **DepthPerspective** frame simultaneously from AirSim
2. Runs `model.track(frame, persist=True)` — YOLOv8 + ByteTrack in one call
3. If no target is selected → streams the YOLO-annotated frame with all boxes
4. If a target **is** selected → overlays a red box + depth readout on the annotated frame and updates `TARGET_CENTER_PX` and `TARGET_DEPTH_M`

### 2. Autonomous Control Loop (10 Hz)
The background thread `autonomous_navigation_thread()`:
1. Reads `TARGET_CENTER_PX` (image-space target position)
2. Calculates horizontal error: `err_x = (cx - frame_center) / frame_half_width`
3. Feeds error into **Steering PID** → clamped to `[-1.0, 1.0]`
4. Derives throttle from base value minus turn penalty, clamped to `[0.35, 0.65]`
5. Falls back to **world-space navigation** using `TARGET_POSITION_3D` if image data is stale
6. Sends `CarControls(throttle, steering)` to AirSim via thread-safe lock

### 3. Object Selection
Clicking on the video triggers a two-stage resolution:
- **Client-side**: detections fetched from `/get_current_detections`, boxes scaled to display coordinates, nearest inside box selected → `/select_track` called with the track ID (avoids coordinate mismatch)
- **Server-side fallback**: if no local hit, scaled coordinates sent to `/select_object` which does a strict then margin-expanded bounding-box hit test

### 4. PID Controllers

```
steering_pid  →  Kp=0.5, Ki=0.0, Kd=0.1   (image-space horizontal error)
throttle_pid  →  Kp=0.3, Ki=0.0, Kd=0.05  (speed error, world-space fallback only)
```

### 5. 3D Coordinate Estimation

```
focal_length = (image_width / 2) / tan(FOV_deg / 2)
x_cam = (pixel_x - cx) / focal_length × depth
y_cam = (pixel_y - cy) / focal_length × depth
→ transform using car pose quaternion → world (x, y, z)
```

---

## 🛠️ Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.9 |
| Unreal Engine + AirSim | Latest Blocks environment |
| CUDA (optional) | For GPU inference acceleration |

---

## 🚀 Setup & Running

### 1. Clone the repository
```powershell
git clone https://github.com/AradhyaChhabdi/DC.git
cd DC
```

### 2. Create and activate a virtual environment
```powershell
python -m venv myenv
.\myenv\Scripts\Activate.ps1
```

### 3. Install dependencies
```powershell
pip install -r requirements.txt
```

### 4. Set the Flask secret key (recommended)
```powershell
$env:FLASK_SECRET_KEY = "your-random-secret-key-here"
```

### 5. Launch AirSim
Open the **Blocks** environment in Unreal Engine and press **Play**. Ensure a `settings.json` with at least one car and a depth camera is configured.

### 6. Start the Flask server
```powershell
cd yolo_flask_app
python app.py
```

### 7. Open the browser
Navigate to **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 🎮 Controls

| Input | Action |
|---|---|
| `↑` / `W` | Drive forward |
| `↓` / `S` | Drive backward |
| `←` / `A` | Turn left |
| `→` / `D` | Turn right |
| `Space` | Brake |
| **Click on object** | Lock target → engage AUTO mode |
| **🎮 button** | Toggle between browser control and AirSim keyboard |

> **AUTO mode** is exited automatically on collision or by clicking **Reset** in the status bar.

---

## 🔌 REST API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serve the main UI |
| `/video_feed` | GET | MJPEG stream of annotated frames |
| `/select_object` | POST | Select target by pixel coordinates `{x, y}` |
| `/select_track` | POST | Select target directly by track ID `{track_id}` |
| `/reset_selection` | POST | Clear target, switch to MANUAL mode |
| `/manual_control` | POST | Send a drive action `{action}` |
| `/toggle_control_source` | POST | Switch API / simulation keyboard control |
| `/get_mode` | GET | Returns current mode, collision flag, arming status |
| `/get_frame_size` | GET | Returns actual frame `{width, height}` |
| `/get_current_detections` | GET | Returns list of `{id, box}` detections |

---

## 🐛 Bugs Fixed

| # | File | Issue | Fix |
|---|---|---|---|
| 1 | `templates/processing.html` | Entire file wrapped in `<!-- -->` — was completely dead HTML | Removed HTML comment wrapper; file is now a valid, functional template |
| 2 | `templates/processing.html` | JS-style `//` comments used inline in HTML body (outside `<script>`) — rendered as visible text | Removed invalid inline comments |
| 3 | `app.py` | `annotated_frame = frame.copy()` when target selected — YOLO annotations for all other objects disappeared | Changed to `results[0].plot()` so non-tracked detections remain visible |
| 4 | `app.py` | `app.secret_key` hardcoded as plain text | Now reads `FLASK_SECRET_KEY` env var with a warning fallback |
| 5 | `templates/index.html` | `best.prefer` key access in click handler, but `prefer` was never stored in `best` dict — silent comparison bug | Added `prefer` to the `best` dict: `best = { id, dist, prefer }` |

---

## 🔮 Potential Future Enhancements

- **Kalman filter** for smoother target state estimation under occlusion
- **Path planning** (A\* / RRT) to route around obstacles while pursuing the target
- **Multiple target queuing** with priority ordering
- **End-to-end deep learning** control policy (replaces PID)
- **Real hardware deployment** via ROS bridge

---

## 📚 Tech Stack

- **[Flask](https://flask.palletsprojects.com/)** — Lightweight Python web framework
- **[Ultralytics YOLOv8](https://docs.ultralytics.com/)** — Object detection + ByteTrack multi-object tracking
- **[OpenCV](https://opencv.org/)** — Frame encoding and annotation
- **[Microsoft AirSim](https://microsoft.github.io/AirSim/)** — Unreal Engine simulation API
- **[NumPy](https://numpy.org/)** — Numerical operations

---

## 📄 License

This project is for academic and research purposes.
