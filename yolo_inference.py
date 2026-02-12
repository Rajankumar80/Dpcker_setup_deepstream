import os
import json
import time
from .config.paths import CONFIG_PATH, DB_PATH, GENDER_LABELS, GENDER_MODEL, INPUT_DIR, PROJECT_ROOT, REID_MODEL, YOLO_MODEL
import cv2
from .database.db import normalize_event
from .database.db import init_db
import numpy as np
import cvzone
import sqlite3
from datetime import date, datetime
from queue import Queue
import threading
from ultralytics import YOLO
from boxmot import StrongSort
from insightface.app import FaceAnalysis
import torch
torch.set_num_threads(4)

# assert torch.cuda.is_available(), "CUDA NOT AVAILABLE"
# video capture logic in frame wise
class FrameStream:
    def __init__(self, folder):
        self.frames = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".png"))
        ])
        self.i = 0

    def read(self):
        if self.i >= len(self.frames):
            return False, None

        img = cv2.imread(self.frames[self.i])
        self.i += 1
        return True, img

    def release(self):
        pass
 
# ===============================
class RetailAnalytics:
    def __init__(self):
        init_db()

        with open(CONFIG_PATH, "r") as f:
            cfg = json.load(f)

        # self.device = "cuda"
        self.device = "cpu"
        self.cameras = cfg["cameras"]
        self.billing_dwell = cfg.get("billing_dwell_time", 10)
        self.log_interval = cfg.get("log_interval", 10)

        # self.detector = YOLO(YOLO_MODEL).to("cuda")
        self.detector = YOLO(YOLO_MODEL)
        # self.detector.to("cpu")


        # self.tracker = StrongSort(
        #     reid_weights=REID_MODEL,
        #     device=0,
        #     half=False,
        #     max_age=180,     #  keep track alive for ~2 sec @30fps
        #     n_init=3, 
        #     iou_threshold=0.15 ,  # supermarket-friendly
        #     appearance_weight=0.85,  #  trust ReID
        #     motion_weight=0.15      # reduce motion dominance

        # )
        self.tracker = StrongSort(
            reid_weights=REID_MODEL,
            device="cpu",
            half=False,
            max_age=180,
            max_unmatched_preds=30,
            n_init=3,
            iou_threshold=0.15,
            appearance_weight=0.85,
            motion_weight=0.15
        )


        self.gender_net = cv2.dnn.readNetFromONNX(GENDER_MODEL)

        # self.face_model = FaceAnalysis(name="buffalo_l")
        # self.face_model.prepare(ctx_id=0, det_size=(640,640))
        self.face_model = FaceAnalysis(name="buffalo_s")  # lighter model
        self.face_model.prepare(ctx_id=-1, det_size=(320,320))

        # Multi-camera setup
        self.captures = {}
        self.current_camera = None
        self.cam_zones = {}
        self.video_name = None
        self.resolution = None

        # Load zones from runtime file (UI-drawn zones)
        runtime_zone_path = os.path.join(PROJECT_ROOT, "zones_runtime.json")
        self.zones = {}
        if os.path.exists(runtime_zone_path):
            try:
                with open(runtime_zone_path, "r") as f:
                    self.zones = json.load(f)
            except Exception as e:
                print("⚠️ Failed to load runtime zones:", e)

        # Initialize first camera
        if self.cameras:
            self.current_camera = list(self.cameras.keys())[0]
            self._open_video()

        self.track_history = {}
        # ⏱️ ZONE MONITORING
        self.cashier_last_seen = time.time()
        self.security_last_seen = time.time()

        self.CASHIER_TIMEOUT = 20    # seconds
        self.SECURITY_TIMEOUT = 20   # seconds
        self.cashier_alert_sent = False
        self.security_alert_sent = False
        self.security_alert_sent = False
        self.unique_tracks = set()
        self.seen_once = set()

        # Track memory
        self.last_positions = {}   # tid → (x,y,time)
        # self.target_fps = 8
        # self.frame_delay = 1.0 / self.target_fps
        self.last_frame_time = 0
        self.staff_pids = set()
        self.load_staff_pids()

        # 🔴 BBOX STABILIZATION STATE
        self.bbox_state = {}
        self.last_seen = {}
        # 🔁 ID SWITCH MONITORING
        # 🧾 BILLING CONVERSION STATE
        self.billing_enter_time = {}   # tid -> timestamp
        self.billing_converted = set() # tids already converted
        # Add in __init__
            # tid -> physical_id
        self.next_pid = 1
        self.last_seen_pid = {}
        self.interactions = {}  # (staff_pid, customer_pid) -> start_time
        self.interaction_events = []
        self.logged_interactions = set()
        self.gender_stats = {"Male": 0, "Female": 0,"unknown":0}
        self.age_stats = {
            "0-10": 0, "10-20": 0, "20-30": 0,
            "30-40": 0, "40-50": 0, "50-60": 0, "60-90": 0,"unknown": 0
        }

        self.events = []
        self.last_dump = time.time()

        self.active_interactions = {}  
        # pid -> {"customer_pid": x, "duration": y}
        self.event_queue = Queue(maxsize=5000)
        self.face_queue = Queue(maxsize=2000)
        self.face_frame_skip = 5
        self.frame_count = 0
        # PID state machine
        self.ENTRY_MAX_WAIT = 3.0    # seconds to wait for face

        # ===============================
        # 🧠 IDENTITY ENGINE (Layer 2)
        # ===============================

        self.tid_to_pid = {}
        self.pid_embeddings = {}      # pid -> embedding
        self.pid_demographics = {}    # pid -> (gender, age)
        self.pid_state = {}           # pid -> TRACK_ONLY / LOCKED

        self.next_pid = 1
        
        # 🔴 FIX B: Load embeddings from database for persistent biometric IDs
        self.load_pid_embeddings()

        # ===============================
        # 🧾 VISIT ENGINE
        # ===============================

        self.active_visits = {}       # pid -> visit_id
        self.visit_start_time = {}    # pid -> entry_time
        self.visit_zones = {}         # visit_id -> {zone: {"enter_time": ts, "inside": bool}}
        self.next_visit_id = 1

        # Zone dwell tracking
        self.zone_dwell = {}          # visit_id -> {zone: dwell_seconds}
        # 🔥 STEP 1 — Create Frame Queue
        self.frame_queue = Queue(maxsize=15)  # 2 seconds buffer at 30 FPS
        
        # Capture metrics
        self.capture_count = 0
        self.capture_drops = 0
        self.capture_failures = 0
        self.capture_second = 0
        
        # Processing metrics
        self.process_count = 0
        self.process_second = 0
        
        # Frame counting and performance monitoring
        self.total_frames = 0
        self.processed_frames = 0
        self.processing_backlog = 0
        
        # FPS calculation variables
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0
        
        # Performance monitoring (per-second counters)
        self.total_latency = 0
        self.max_latency = 0
        self.latency_second = 0
        self.last_stats_time = time.time()
        self.second_processed = 0
        self.second_dropped = 0
        
        # Frame counter for face skip logic
        self.frame_count = 0
        
        # 🔥 SHUTDOWN FLAG MUST BE DEFINED BEFORE THREADS
        self.shutdown = False
        
        # Gate and interaction state
        self.pid_gate_side = {}
        self.pid_gate_last_event = {}
        self.GATE_COOLDOWN = 2.0  # seconds
        self.pid_last_vote_time = {}
        self.VOTE_COOLDOWN = 1.0  # seconds
        
        # 🔥 START DB WORKER
        threading.Thread(target=self.db_worker, daemon=True).start()
        
        # Start threads
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
        self.capture_thread.start()
        self.processing_thread.start()



    # def _open_video(self):
    #     print("Start video value:", self.start_video)
        
    #     # Check if we should use RTSP camera
    #     if self.start_video == "rtsp":
    #         print("🎥 Opening RTSP camera...")
    #         self.cap = cv2.VideoCapture("")
    #         self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    #         self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    #         self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffering for live stream

    #         if not self.cap.isOpened():
    #             raise RuntimeError("❌ RTSP camera not accessible")

    #         self.video_name = "rtsp_camera"
    #         self.resolution = None

    #         # Load zones from runtime file (UI-drawn zones)
    #         runtime_zone_path = os.path.join(PROJECT_ROOT, "zones_runtime.json")

    #         self.gate = None
    #         self.billing_zone = None
    #         self.cashier_zone = None
    #         self.security_zone = None

    #         if os.path.exists(runtime_zone_path):
    #             try:
    #                 with open(runtime_zone_path, "r") as f:
    #                     runtime = json.load(f)

    #                 self.gate = runtime.get("gate_line")
    #                 self.billing_zone = runtime.get("billing")
    #                 self.cashier_zone = runtime.get("cashier")
    #                 self.security_zone = runtime.get("security")

    #                 print("✅ Zones loaded for RTSP camera")

    #             except Exception as e:
    #                 print("⚠️ Failed to load zones:", e)
    #     else:
    #         # Use file-based video processing
    #         self.video_name = self.videos[0]

    #         # -------------------------------
    #         # Load per-video config
    #         # -------------------------------
    #         meta = self.video_cfg.get(self.video_name, {})
    #         self.resolution = meta.get("resolution", None)
    #         self.zones = meta.get("zones", {})

    #         # -------------------------------
    #         # Runtime zones override (UI > config)
    #         # -------------------------------
    #         runtime_zone_path = os.path.join(PROJECT_ROOT, "zones_runtime.json")
    #         runtime = None

    #         if os.path.exists(runtime_zone_path):
    #             try:
    #                 with open(runtime_zone_path, "r") as f:
    #                     runtime = json.load(f)
    #             except Exception as e:
    #                 print("⚠️ Failed to load runtime zones:", e)

    #         if runtime and runtime.get("video") in (None, self.video_name):
    #             self.gate = runtime.get("gate_line")
    #             self.billing_zone = runtime.get("billing")
    #             self.cashier_zone = runtime.get("cashier")
    #             self.security_zone = runtime.get("security")
    #         else:
    #             self.gate = self.zones.get("gate_line")
    #             self.billing_zone = self.zones.get("billing")
    #             self.cashier_zone = self.zones.get("cashier")
    #             self.security_zone = self.zones.get("security")

    #         # -------------------------------
    #         # Open video / frame stream (ALWAYS)
    #         # -------------------------------
    #         path = os.path.join(INPUT_DIR, self.video_name)

    #         if os.path.isdir(path):
    #             self.cap = FrameStream(path)
    #         else:
    #             self.cap = cv2.VideoCapture(path)
    def _open_video(self):
        print("Current camera:", self.current_camera)

        # Multi-camera RTSP setup
        if self.current_camera in self.cameras:
            cam_info = self.cameras[self.current_camera]
            rtsp_url = cam_info["rtsp"]
            self.resolution = cam_info.get("resolution", None)
            self.video_name = self.current_camera

            print(f"🎥 Opening RTSP camera: {self.current_camera}")

            self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

            # Reduce latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if self.resolution:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

            if not self.cap.isOpened():
                raise RuntimeError(f"❌ RTSP camera {self.current_camera} not accessible")

            # Load camera-specific zones
            if self.current_camera in self.zones:
                self.cam_zones = self.zones[self.current_camera]
            else:
                # fallback for flat structure
                self.cam_zones = self.zones
            
            self.gate = self.cam_zones.get("gate_line")
            self.billing_zone = self.cam_zones.get("billing")
            self.cashier_zone = self.cam_zones.get("cashier")
            self.security_zone = self.cam_zones.get("security")

            print(f"✅ Zones loaded for camera: {self.current_camera}")

        else:
            # File mode fallback
            print("🎞 Opening file-based video...")

            self.video_name = self.start_video

            meta = self.video_cfg.get(self.video_name, {})
            self.resolution = meta.get("resolution", None)
            self.zones = meta.get("zones", {})

            runtime_zone_path = os.path.join(PROJECT_ROOT, "zones_runtime.json")
            runtime = None

            if os.path.exists(runtime_zone_path):
                try:
                    with open(runtime_zone_path, "r") as f:
                        runtime = json.load(f)
                except:
                    pass

            if runtime and runtime.get("video") in (None, self.video_name):
                self.gate = runtime.get("gate_line")
                self.billing_zone = runtime.get("billing")
                self.cashier_zone = runtime.get("cashier")
                self.security_zone = runtime.get("security")
            else:
                self.gate = self.zones.get("gate_line")
                self.billing_zone = self.zones.get("billing")
                self.cashier_zone = self.zones.get("cashier")
                self.security_zone = self.zones.get("security")

            path = os.path.join(INPUT_DIR, self.video_name)
            self.cap = cv2.VideoCapture(path)

        


    def load_staff_pids(self):
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT pid FROM staff_registry")
        rows = cur.fetchall()
        self.staff_pids = {row[0] for row in rows}
        conn.close()

    def load_pid_embeddings(self):
        """🔴 FIX B: Load embeddings from database for persistent biometric IDs"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            
            # Load face embeddings from faces table
            cur.execute("""
                SELECT DISTINCT pid, embedding 
                FROM faces 
                WHERE embedding IS NOT NULL
            """)
            rows = cur.fetchall()
            
            for pid, embedding_blob in rows:
                try:
                    # Convert blob back to numpy array
                    if embedding_blob:
                        embedding_array = np.frombuffer(embedding_blob, dtype=np.float32)
                        if pid not in self.pid_embeddings:
                            self.pid_embeddings[pid] = embedding_array
                            self.pid_state[pid] = "FACE_CONFIRMED"
                except Exception as e:
                    print(f"Error loading embedding for PID {pid}: {e}")
                    continue
            
            # Also load demographics from shopper_profiles
            cur.execute("SELECT pid, gender, age_group FROM shopper_profiles")
            profile_rows = cur.fetchall()
            for pid, gender, age_group in profile_rows:
                if pid not in self.pid_demographics:
                    self.pid_demographics[pid] = (gender, age_group)
                    self.pid_state[pid] = "LOCKED"
            
            # Find the highest PID to set next_pid correctly
            cur.execute("SELECT MAX(pid) FROM faces UNION SELECT MAX(pid) FROM shopper_profiles")
            max_pid_result = cur.fetchone()
            if max_pid_result and max_pid_result[0]:
                self.next_pid = max_pid_result[0] + 1
            
            conn.close()
            print(f"✅ Loaded {len(self.pid_embeddings)} persistent PIDs with embeddings from database")
            
        except Exception as e:
            print(f"⚠️ Failed to load PID embeddings: {e}")
            

    def classify_gender(self, crop):
        blob = cv2.dnn.blobFromImage(
            crop, 1.0, (227, 227),
            (78.426, 87.768, 114.895),
            swapRB=False
        )
        self.gender_net.setInput(blob)
        return GENDER_LABELS[self.gender_net.forward().argmax()]

    def classify_age(self, h, frame_h):
        r = h / frame_h
        if r < 0.15: return "0-10"
        elif r < 0.22: return "10-20"
        elif r < 0.28: return "20-30"
        elif r < 0.34: return "30-40"
        elif r < 0.40: return "40-50"
        elif r < 0.46: return "50-60"
        else: return "60-90"
    def point_in_zone(self, cx, cy, zone):
        if not zone:
            return False
        return cv2.pointPolygonTest(
            np.array(zone, np.int32),
            (cx, cy),
            False
        ) >= 0
    def get_face(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        person_crop = frame[y1:y2, x1:x2]

        if person_crop.size == 0:
            return None, None

        faces = self.face_model.get(person_crop)
        if len(faces) == 0:
            return None, None

        # f = faces[0]
        f = max(faces, key=lambda x: x.det_score)

        if f.det_score < 0.6:
            return None, None


        fx1, fy1, fx2, fy2 = map(int, f.bbox)

        # 🔐 Clamp face box to image boundaries
        h, w = person_crop.shape[:2]
        fx1 = max(0, min(fx1, w-1))
        fx2 = max(0, min(fx2, w-1))
        fy1 = max(0, min(fy1, h-1))
        fy2 = max(0, min(fy2, h-1))

        if fx2 <= fx1 or fy2 <= fy1:
            return None, None

        face_img = person_crop[fy1:fy2, fx1:fx2]

        if face_img.size == 0:
            return None, None

        return f.embedding, face_img
    def get_role(self, pid):
        return "STAFF" if pid in self.staff_pids else "CUSTOMER"
    def match_face(self, emb):
        best_pid = None
        best_sim = 0

        for pid, db_emb in self.pid_embeddings.items():
            sim = np.dot(emb, db_emb) / (np.linalg.norm(emb) * np.linalg.norm(db_emb))

            if sim > best_sim:
                best_sim = sim
                best_pid = pid

        # threshold
        if best_sim > 0.55:
            return best_pid
        return None


    def inherit_pid(self, tid, cx, cy):
        now = time.time()
        best_pid = None
        best_dist = 1e9

        for pid, data in self.last_seen_pid.items():
            if data is None:
                continue

            px, py, ts = data
            if now - ts > 2.0:
                continue

            d = np.hypot(cx - px, cy - py)
            if d < best_dist and d < 80:
                best_dist = d
                best_pid = pid

        return best_pid
    # ===============================
    # ✅ FIXED BBOX FUNCTIONS
    # ===============================
    def occlusion_guard(self, tid, bbox):
        if tid not in self.bbox_state:
            return bbox

        prev = self.bbox_state[tid]
        prev_h = prev[3] - prev[1]
        curr_h = bbox[3] - bbox[1]

        if curr_h < 0.5 * prev_h:
            return prev

        return bbox

    def smooth_bbox(self, tid, bbox, h, frame_h):
        ratio = h / frame_h

        if ratio > 0.4:
            alpha = 0.85
        elif ratio > 0.25:
            alpha = 0.70
        else:
            alpha = 0.45

        if tid not in self.bbox_state:
            self.bbox_state[tid] = bbox
            return bbox

        prev = self.bbox_state[tid]

        smoothed = [
            int(alpha * prev[i] + (1 - alpha) * bbox[i])
            for i in range(4)
        ]

        self.bbox_state[tid] = smoothed
        return smoothed
    def distance(self, p1, p2):
        return np.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def db_worker(self):
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cur = conn.cursor()

        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA temp_store=MEMORY;")

        face_batch = []
        last_flush = time.time()

        while not self.shutdown:
            try:
                try:
                    face_batch.append(self.face_queue.get_nowait())
                except:
                    pass

                if len(face_batch) >= 5:
                    cur.executemany("""
                        INSERT INTO faces
                        (pid, face, embedding, date, time, video)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, face_batch)
                    conn.commit()
                    face_batch.clear()
        
                # time-based flush
                if time.time() - last_flush > 1.0:
                    if face_batch:
                        cur.executemany("""
                            INSERT INTO faces
                            (pid, face, embedding, date, time, video)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, face_batch)
                        conn.commit()
                        face_batch.clear()

                    last_flush = time.time()

                time.sleep(0.01)

            except Exception as e:
                print("DB WORKER ERROR:", e)
        self.shutdown = True
        time.sleep(1.5)
        conn.close()
        
    # ===============================
    def handle_entry(self, pid):
        if pid in self.active_visits:
            return

        visit_id = f"V{self.next_visit_id}"
        self.next_visit_id += 1

        self.active_visits[pid] = visit_id
        self.visit_start_time[pid] = datetime.now()
        self.visit_zones[visit_id] = {}
        self.zone_dwell[visit_id] = {}

        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO visits (visit_id, pid, entry_time, video)
                VALUES (?, ?, ?, ?)
            """, (
                visit_id,
                pid,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                self.video_name
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            print("Visit insert error:", e)

    def handle_zone(self, pid, zone_name, currently_inside):
        if pid not in self.active_visits:
            return

        visit_id = self.active_visits[pid]
        now = time.time()

        # Initialize zone state if not exists
        if zone_name not in self.visit_zones[visit_id]:
            if currently_inside:
                self.visit_zones[visit_id][zone_name] = {
                    "enter_time": now,
                    "inside": True
                }
            return

        zone_state = self.visit_zones[visit_id][zone_name]
        previously_inside = zone_state["inside"]
        
        # State transition logic
        if currently_inside and not previously_inside:
            # Person entered the zone
            zone_state["enter_time"] = now
            zone_state["inside"] = True
        elif not currently_inside and previously_inside:
            # Person left the zone - compute dwell and insert
            dwell = now - zone_state["enter_time"]
            zone_state["inside"] = False
            
            # Persist zone exit event
            try:
                conn = sqlite3.connect(DB_PATH)
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO zone_events
                    (visit_id, zone, dwell_time)
                    VALUES (?, ?, ?)
                """, (
                    visit_id,
                    zone_name,
                    round(dwell, 2)
                ))
                conn.commit()
                conn.close()
            except Exception as e:
                print("Zone exit insert error:", e)

    def billing_validate(self, pid, embedding):
        if pid not in self.active_visits:
            return pid

        visit_id = self.active_visits[pid]

        matched_pid = self.match_face(embedding)

        if matched_pid is None:
            return pid

        if matched_pid == pid:
            return pid

        # 🚨 Identity mismatch detected
        print(f"⚠ PID CORRECTION: {pid} → {matched_pid}")

        old_pid = pid
        new_pid = matched_pid

        # Update visit mapping
        self.active_visits[new_pid] = visit_id
        del self.active_visits[old_pid]

                # Update DB visit record
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("""
                UPDATE visits
                SET pid = ?
                WHERE visit_id = ?
            """, (new_pid, visit_id))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print("Correction DB error:", e)

        return new_pid

    def close_visit(self, pid):
        if pid not in self.active_visits:
            return

        visit_id = self.active_visits[pid]

        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("""
                UPDATE visits
                SET exit_time = ?
                WHERE visit_id = ?
            """, (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                visit_id
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            print("Visit close error:", e)

        del self.active_visits[pid]

    def run(self):
        """Main execution loop - UI display and metrics only"""
        while True:
            # Display real-time statistics on frame
            current_time = time.time()
            if current_time - self.fps_start_time >= 1.0:
                self.current_fps = self.fps_frame_count / (current_time - self.fps_start_time)
                self.fps_frame_count = 0
                self.fps_start_time = current_time

            # Create a blank frame for displaying metrics when no processing is happening
            # or use the last processed frame if available
            if hasattr(self, '_last_frame') and self._last_frame is not None:
                display_frame = self._last_frame.copy()
            else:
                # Create a black frame with text overlay
                display_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    display_frame,
                    "WAITING FOR PROCESSING...",
                    (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    3
                )
            
            # Display real-time statistics on frame
            cv2.putText(
                display_frame,
                f"FPS: {self.current_fps:.1f}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
            
            cv2.putText(
                display_frame,
                f"Total: {self.total_frames}",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            cv2.putText(
                display_frame,
                f"Capture Drops: {self.capture_drops}",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            
            drop_rate = (self.capture_drops / self.capture_count * 100) if self.capture_count > 0 else 0
            cv2.putText(
                display_frame,
                f"Drop Rate: {drop_rate:.1f}%",
                (20, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            # Print detailed statistics every second
            if current_time - self.last_stats_time >= 1.0:
                # Calculate per-second metrics (accurate latency averaging)
                per_second_processed = self.second_processed
                per_second_dropped = self.second_dropped
                per_second_total = per_second_processed + per_second_dropped
                
                avg_latency = (self.total_latency / per_second_processed) if per_second_processed > 0 else 0
                processing_efficiency = (per_second_processed / per_second_total * 100) if per_second_total > 0 else 0
                
                print(f"\n📊 FRAME PROCESSING STATS")
                print(f"Total Frames: {self.total_frames}")
                print(f"Processed: {self.processed_frames}")
                print(f"Capture Drops: {self.capture_drops}")
                print(f"Per-Second Efficiency: {processing_efficiency:.1f}%")
                print(f"Avg Processing Time: {avg_latency*1000:.1f} ms")
                print(f"Max Processing Time: {self.max_latency*1000:.1f} ms")
                
                # Reset per-second counters for next second
                self.total_latency = 0
                self.max_latency = 0
                self.second_processed = 0
                self.second_dropped = 0
                self.last_stats_time = current_time

            cv2.imshow("Retail Analytics", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.shutdown = True
                time.sleep(1.5)  # allow db worker to flush
                break

            # Reduce CPU spinning in main thread
            time.sleep(0.01)

    def capture_loop(self):
        """Thread 1 → Capture (constant rate)"""
        print("Capture running")

        if self.current_camera not in self.cameras:
            target_fps = 30
            frame_delay = 1.0 / target_fps
        else:
            frame_delay = None
        
        while not self.shutdown:
            start = time.time()
            ret, frame = self.cap.read()
            now = time.time()

            if not ret:
                self.capture_failures += 1
                continue

            self.capture_count += 1
            self.capture_second += 1
            self.total_frames += 1

            try:
                self.frame_queue.put_nowait((frame, now))
            except:
                # Queue full → real frame drop
                self.capture_drops += 1
                self.second_dropped += 1

            # Throttle capture to prevent overwhelming processing thread (only for file processing)
            if frame_delay:
                elapsed = time.time() - start
                sleep_time = frame_delay - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def processing_loop(self):
        """Thread 2 → Processing (YOLO pipeline)"""
        while not self.shutdown:
            try:
                frame, capture_time = self.frame_queue.get(timeout=1)
            except:
                continue

            process_start = time.time()

            # resize only valid frame
            if self.resolution:
                frame = cv2.resize(frame, tuple(self.resolution))

            frame_h = frame.shape[0]

            # ===============================
            # 1️⃣ YOLO DETECTION
            # ===============================
            results = self.detector.predict(
                frame,
                conf=0.2,
                imgsz=320,
                device="cpu",
                verbose=False
            )[0]

            # ===============================
            # 2️⃣ BUILD DETECTIONS
            # ===============================
            dets = []
            frame_h = frame.shape[0]

            for b in results.boxes:
                conf = float(b.conf[0])
                cls = int(b.cls[0])

                if cls != 0:
                    continue

                x1, y1, x2, y2 = map(int, b.xyxy[0])
                h = y2 - y1
                if h < 0.08* frame_h:
                    continue

                # StrongSort expects format: [x1, y1, x2, y2, confidence, class]
                dets.append([x1, y1, x2, y2, conf, cls])

            # ===============================
            # 3️⃣ BYTE TRACKING
            # ===============================
            if len(dets) > 0:
                tracks_raw = self.tracker.update(
                    np.array(dets, dtype=np.float32),
                    frame
                )
            else:
                tracks_raw = []

            tracks = []
            for t in tracks_raw:
                x1, y1, x2, y2, tid = map(int, t[:5])
                tracks.append([x1, y1, x2, y2, tid])

            # ===============================
            # 4️⃣ TRACK PROCESSING
            # ===============================
            for t in tracks:
                face_img = None 
                face_emb = None  # 🔴 FIX A: Initialize face_emb to None
                x1, y1, x2, y2, tid = map(int, t[:5])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                h = y2 - y1  # 🔥 MOVE h CALCULATION HERE

                bbox = [x1, y1, x2, y2]
                now_ts = time.time()

                # ✅ UPDATE POSITION FIRST
                self.last_positions[tid] = (cx, cy, now_ts)

                # 🧠 CLEAN OLD POSITIONS
                for old_tid in list(self.last_positions.keys()):
                    if now_ts - self.last_positions[old_tid][2] > 2.0:
                        self.last_positions.pop(old_tid, None)

                # ==============================
                # 🔐 FACE + PID ASSIGNMENT (HYBRID)
                if tid in self.tid_to_pid:
                    pid = self.tid_to_pid[tid]
                else:
                    face_emb, face_img = (None, None)
                    if self.frame_count % self.face_frame_skip == 0:
                        face_emb, face_img = self.get_face(frame, bbox)

                    if face_emb is not None:
                        matched_pid = self.match_face(face_emb)
                        if matched_pid is not None:
                            pid = matched_pid
                        else:
                            pid = self.next_pid
                            self.next_pid += 1

                        if pid not in self.pid_embeddings:
                            self.pid_embeddings[pid] = face_emb

                        prev = self.pid_state.get(pid, "TRACK_ONLY")
                        if prev != "DEMO_LOCKED":
                            self.pid_state[pid] = "FACE_CONFIRMED"

                        if face_img is not None:
                            success, buffer = cv2.imencode(".png", face_img)
                            if success:
                                now = datetime.now()
                                try:
                                    self.face_queue.put_nowait((
                                        pid,
                                        buffer.tobytes(),
                                        face_emb.astype(np.float32).tobytes() if face_emb is not None else None,
                                        now.strftime("%Y-%m-%d"),
                                        now.strftime("%H:%M:%S"),
                                        self.video_name
                                    ))
                                except:
                                    pass
                    else:
                        pid = self.inherit_pid(tid, cx, cy)
                        if pid is None:
                            pid = self.next_pid
                            self.next_pid += 1

                        if pid not in self.pid_state:
                            self.pid_state[pid] = "TRACK_ONLY"

                    self.tid_to_pid[tid] = pid

                # ===============================
                # 🧾 FINALIZE ENTRY (SMART DEMOGRAPHICS)
                # ===============================
                # No longer needed - visit creation IS entry now

                # 👤 DEMOGRAPHIC LOCK — SINGLE GOOD FACE WINS
                if face_img is not None and pid not in self.pid_demographics:
                    gender = self.classify_gender(face_img)
                    age = self.classify_age(h, frame_h)  # Now h is defined!

                    self.pid_demographics[pid] = (gender, age)
                    self.pid_state[pid] = "LOCKED"

                    try:
                        conn = sqlite3.connect(DB_PATH)
                        cur = conn.cursor()
                        cur.execute("""
                            INSERT OR IGNORE INTO shopper_profiles
                            (pid, gender, age_group)
                            VALUES (?, ?, ?)
                        """, (pid, gender, age))
                        conn.commit()
                        conn.close()
                    except:
                        pass

                role = self.get_role(pid)
                self.last_seen_pid[pid] = (cx, cy, time.time())

                # 👥 STAFF–CUSTOMER INTERACTION
                if role == "STAFF":
                    for other_tid, (ox, oy, ots) in self.last_positions.items():
                        other_pid = self.tid_to_pid.get(other_tid)
                        if other_pid is None or other_pid == pid:
                            continue
                        if other_pid in self.staff_pids:
                            continue

                        d = self.distance((cx, cy), (ox, oy))
                        key = (pid, other_pid)

                        if d < 80:
                            if key not in self.interactions:
                                self.interactions[key] = time.time()
                            else:
                                duration = time.time() - self.interactions[key]
                                self.active_interactions[pid] = {
                                    "customer_pid": other_pid,
                                    "duration": duration
                                }

                                if duration >= 2 and key not in self.logged_interactions:
                                    self.logged_interactions.add(key)
                                    now = datetime.now()
                                    try:
                                        conn = sqlite3.connect(DB_PATH)
                                        cur = conn.cursor()
                                        cur.execute("""
                                            INSERT INTO interactions
                                            (date, time, staff_pid, customer_pid, duration, video)
                                            VALUES (?, ?, ?, ?, ?, ?)
                                        """, (
                                            now.strftime("%Y-%m-%d"),
                                            now.strftime("%H:%M:%S"),
                                            pid,
                                            other_pid,
                                            round(duration, 2),
                                            self.video_name
                                        ))
                                        conn.commit()
                                        conn.close()
                                    except Exception as e:
                                        print("❌ Interaction DB error:", e)
                        else:
                            self.interactions.pop(key, None)
                            self.logged_interactions.discard(key)
                            self.active_interactions.pop(pid, None)

                raw_bbox = [x1, y1, x2, y2]
                bbox = self.occlusion_guard(tid, raw_bbox)
                bbox = self.smooth_bbox(tid, bbox, h, frame_h)

                x1, y1, x2, y2 = bbox
                self.last_seen[tid] = time.time()
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                now_ts = time.time()

                # CASHIER PRESENCE
                if self.cashier_zone:
                    currently_inside = self.point_in_zone(cx, cy, self.cashier_zone)
                    self.handle_zone(pid, "cashier", currently_inside)
                    if currently_inside:
                        self.cashier_last_seen = time.time()
                        self.cashier_alert_sent = False

                # SECURITY PRESENCE
                if self.security_zone:
                    currently_inside = self.point_in_zone(cx, cy, self.security_zone)
                    self.handle_zone(pid, "security", currently_inside)
                    if currently_inside:
                        self.security_last_seen = time.time()
                        self.security_alert_sent = False

            # 🚪 ENTRY / EXIT GATE
                if self.gate:
                    gate_x = self.gate["p1"][0]
                    curr_side = "LEFT" if cx < gate_x else "RIGHT"

                    if pid not in self.pid_gate_side:
                        self.pid_gate_side[pid] = curr_side
                    else:
                        prev_side = self.pid_gate_side[pid]
                        if prev_side != curr_side:
                            self.pid_gate_side[pid] = curr_side
                            if prev_side == "LEFT":
                                self.handle_entry(pid)
                            elif prev_side == "RIGHT":
                                self.close_visit(pid)

                # 🎨 DARK GREEN / PINK BBOX
                cvzone.cornerRect(
                    frame,
                    (x1, y1, x2 - x1, y2 - y1),
                    l=6,
                    rt=2,
                    colorR=(0, 255, 0)
                )

                label = f"{role} | PID {pid} | TID {tid}"
                if pid in self.pid_demographics:
                    g, a = self.pid_demographics[pid]
                    label += f" | {g} | {a}"

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2
                )

                # 🧾 BILLING CONVERSION LOGIC
                if self.billing_zone:
                    currently_inside = self.point_in_zone(cx, cy, self.billing_zone)
                    self.handle_zone(pid, "billing", currently_inside)
                    
                    # 🚨 BILLING VALIDATION (Identity Authority)
                    if currently_inside and face_emb is not None:
                        corrected_pid = self.billing_validate(pid, face_emb)
                        if corrected_pid != pid:
                            pid = corrected_pid
                            self.tid_to_pid[tid] = pid

                # unique customer
                if self.gate:
                    gate_x = self.gate["p1"][0]
                    ENTRY_MARGIN = 40  # pixels inside store

                    if cx > gate_x + ENTRY_MARGIN:
                        if pid not in self.seen_once:
                            self.seen_once.add(pid)
                            self.unique_tracks.add(f"{self.video_name}_{pid}")

                    self.unique_tracks.add(f"{self.video_name}_{pid}")

                if pid in self.active_interactions:
                    info = self.active_interactions[pid]
                    duration = info["duration"]

                    cv2.putText(
                        frame,
                        f"INTERACTING {int(duration)}s",
                        (cx - 40, cy - 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

            # # 🎨 DRAW ZONES ON FRAME (ADDED)
            # print(f"DEBUG: Drawing zones for camera {self.current_camera}")
            # print(f"DEBUG: Gate: {self.gate}")
            # print(f"DEBUG: Billing: {self.billing_zone}")
            # print(f"DEBUG: Cashier: {self.cashier_zone}")
            # print(f"DEBUG: Security: {self.security_zone}")

            if self.gate and self.gate["p1"] and self.gate["p2"]:
                p1 = tuple(self.gate["p1"])
                p2 = tuple(self.gate["p2"])
                cv2.line(frame, p1, p2, (255, 0, 0), 2)  # Blue line for gate
                cv2.putText(frame, "GATE", (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                # print(f"DEBUG: Drew gate line from {p1} to {p2}")

            if self.billing_zone and len(self.billing_zone) >= 3:
                pts = np.array(self.billing_zone, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True, (0, 255, 255), 2)  # Yellow for billing
                cv2.putText(frame, "BILLING", (self.billing_zone[0][0], self.billing_zone[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                # print(f"DEBUG: Drew billing zone with {len(self.billing_zone)} points")

            if self.cashier_zone and len(self.cashier_zone) >= 3:
                pts = np.array(self.cashier_zone, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True, (255, 255, 0), 2)  # Cyan for cashier
                cv2.putText(frame, "CASHIER", (self.cashier_zone[0][0], self.cashier_zone[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                # print(f"DEBUG: Drew cashier zone with {len(self.cashier_zone)} points")

            if self.security_zone and len(self.security_zone) >= 3:
                pts = np.array(self.security_zone, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True, (0, 0, 255), 2)  # Red for security
                cv2.putText(frame, "SECURITY", (self.security_zone[0][0], self.security_zone[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                # print(f"DEBUG: Drew security zone with {len(self.security_zone)} points")

            # Calculate current FPS for display
            current_time = time.time()
            if current_time - self.fps_start_time >= 1.0:
                self.current_fps = self.fps_frame_count / (current_time - self.fps_start_time)
                self.fps_frame_count = 0
                self.fps_start_time = current_time

            # Display real-time statistics on frame
            cv2.putText(
                frame,
                f"FPS: {self.current_fps:.1f}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
            
            cv2.putText(
                frame,
                f"Total: {self.total_frames}",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            cv2.putText(
                frame,
                f"Capture Drops: {self.capture_drops}",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            
            drop_rate = (self.capture_drops / self.capture_count * 100) if self.capture_count > 0 else 0
            cv2.putText(
                frame,
                f"Drop Rate: {drop_rate:.1f}%",
                (20, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            # Calculate and display processing statistics
            process_end = time.time()
            self.process_count += 1
            self.process_second += 1
            self.frame_count += 1
            self.second_processed += 1
            self.fps_frame_count += 1
            
            lat = process_end - capture_time
            self.latency_second += lat
            self.total_latency += lat
            self.max_latency = max(self.max_latency, lat)

            # Update processed frame counter
            self.processed_frames += 1

            # Save frame for UI display
            self._last_frame = frame

            # Print detailed statistics every second
            if current_time - self.last_stats_time >= 1.0:
                # Calculate per-second metrics
                per_second_processed = self.second_processed
                per_second_dropped = self.second_dropped
                per_second_total = per_second_processed + per_second_dropped
                
                avg_latency = (self.latency_second / per_second_processed) if per_second_processed > 0 else 0
                processing_efficiency = (per_second_processed / per_second_total * 100) if per_second_total > 0 else 0
                
                # 🔥 METRICS LOGIC (THIS IS WHAT YOU ASKED FOR)
                capture_fps = self.capture_second
                processing_fps = self.second_processed
                queue_size = self.frame_queue.qsize()
                drop_rate = (self.capture_drops / self.capture_count * 100) if self.capture_count > 0 else 0
                mismatch = capture_fps - processing_fps
                
                print(f"\n📊 REAL-TIME SYSTEM METRICS")
                print(f"Capture FPS: {capture_fps:.1f}")
                print(f"Processing FPS: {processing_fps:.1f}")
                print(f"Queue Size: {queue_size}")
                print(f"Capture Drops: {self.capture_drops}")
                print(f"Drop Rate: {drop_rate:.1f}%")
                print(f"Avg Latency: {avg_latency*1000:.1f} ms")
                print(f"Max Latency: {self.max_latency*1000:.1f} ms")
                print(f"Mismatch: {mismatch:.1f}")
                
                # 🔥 SYSTEM OVERLOAD DETECTION LOGIC
                if queue_size > 0.8 * self.frame_queue.maxsize:
                    print("⚠️ SYSTEM OVERLOAD")
                elif mismatch > 3:
                    print("⚠️ Processing slower than capture")
                
                # Reset per-second counters for next second
                self.capture_second = 0
                self.process_second = 0
                self.second_processed = 0
                self.second_dropped = 0
                self.latency_second = 0
                self.max_latency = 0
                self.last_stats_time = current_time

            # except Exception as e:
            #       print("PROCESSING THREAD ERROR:", e)
def run():
    print("🔥 THIS FILE IS RUNNING")
    RetailAnalytics().run()     

