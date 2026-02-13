import cv2
import numpy as np
import time
import threading
from queue import Queue
from ultralytics import YOLO
from boxmot import ByteTrack
from openvino.runtime import Core
import os

# Limit OpenVINO threads (IMPORTANT)
os.environ["OMP_NUM_THREADS"] = "4"

# =========================================================
# CONFIG
# =========================================================
CAMERA_URL = "rtsp://10.64.36.13:554/rtsp/streaming?channel=01&subtype=1"

# YOLO_MODEL_PATH = r"C:\Users\TP-User\Desktop\store_analytics_proj\yolov8s_int8_openvino_model"
YOLO_MODEL_PATH = r"C:\Users\TP-User\Desktop\store_analytics_proj\yolov8s.pt"
FACE_DET_PATH = r"C:\Users\TP-User\Desktop\store_analytics_proj\face-detection-retail-0004.xml"
FACE_REID_PATH = r"C:\Users\TP-User\Desktop\store_analytics_proj\face-reidentification-retail-0095.xml"

FACE_THRESHOLD = 0.65
REID_REFRESH_FRAMES = 40

# =========================================================
# QUEUES
# =========================================================
frame_queue = Queue(maxsize=5)       # decode → inference
face_queue = Queue(maxsize=50)       # inference → face worker

# =========================================================
# GLOBAL MEMORY
# =========================================================
pid_embeddings = {}
tid_to_pid = {}
tid_last_reid_frame = {}
next_pid = 1
pid_lock = threading.Lock()


# =========================================================
# OPENVINO INIT
# =========================================================
core = Core()

face_det_model = core.read_model(FACE_DET_PATH)
compiled_face_det = core.compile_model(face_det_model, "CPU")
face_det_output = compiled_face_det.output(0)

face_reid_model = core.read_model(FACE_REID_PATH)
compiled_face_reid = core.compile_model(face_reid_model, "CPU")
face_reid_output = compiled_face_reid.output(0)

# =========================================================
# THREAD 1: RTSP DECODE
# =========================================================
def decode_thread():
    capture_counter = 0
    start_time = time.time()
    cap = cv2.VideoCapture(CAMERA_URL)
    

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        capture_counter += 1

        # Drop frame if queue full (IMPORTANT)
        if frame_queue.full():
            continue

        frame_queue.put(frame)
        # calculate every 1 second
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            capture_fps = capture_counter / elapsed
            print(f"Camera FPS: {capture_fps:.2f}")
            capture_counter = 0
            start_time = time.time()

# =========================================================
# THREAD 3: FACE WORKER (ASYNC)
# =========================================================
def face_worker():
    global next_pid

    while True:
        item = face_queue.get()
        if item is None:
            break

        person_crop, cam_tid, frame_id = item

        resized = cv2.resize(person_crop, (300, 300))
        blob = resized.transpose(2, 0, 1)[None].astype(np.float32)

        detections = compiled_face_det([blob])[face_det_output][0][0]

        for det in detections:
            if det[2] < 0.6:
                continue

            fx1 = int(det[3] * person_crop.shape[1])
            fy1 = int(det[4] * person_crop.shape[0])
            fx2 = int(det[5] * person_crop.shape[1])
            fy2 = int(det[6] * person_crop.shape[0])

            face_crop = person_crop[fy1:fy2, fx1:fx2]
            if face_crop.size == 0:
                continue

            face_crop = cv2.resize(face_crop, (128, 128))
            face_blob = face_crop.transpose(2, 0, 1)[None].astype(np.float32)

            result = compiled_face_reid([face_blob])[face_reid_output]
            emb = result.flatten()
            emb = emb / np.linalg.norm(emb)

            with pid_lock:
                best_pid = None
                best_score = 0

                for pid, stored in pid_embeddings.items():
                    score = np.dot(emb, stored)
                    if score > best_score:
                        best_score = score
                        best_pid = pid

                if best_score > FACE_THRESHOLD:
                    pid = best_pid
                else:
                    pid = next_pid
                    pid_embeddings[pid] = emb
                    next_pid += 1

                tid_to_pid[cam_tid] = pid
                tid_last_reid_frame[cam_tid] = frame_id

            break

# =========================================================
# THREAD 2: INFERENCE + TRACKING
# =========================================================
def inference_thread():

    model = YOLO(YOLO_MODEL_PATH)
    tracker = ByteTrack(track_thresh=0.4, match_thresh=0.8, min_hits=2)

    prev_boxes = {}
    alpha = 0.8
    frame_index = 0

    second_capture = 0
    second_process = 0
    latency_second = 0
    max_latency = 0
    last_stats_time = time.time()

    while True:
        if frame_queue.empty():
            continue

        start = time.time()
        frame = frame_queue.get()
        frame_index += 1
        second_capture += 1

        results = model.predict(frame, conf=0.3, imgsz=320, verbose=False)[0]

        dets = []
        for b in results.boxes:
            if int(b.cls[0]) != 0:
                continue
            x1,y1,x2,y2 = map(int, b.xyxy[0])
            conf = float(b.conf[0])
            dets.append([x1,y1,x2,y2,conf,0])

        tracks = tracker.update(np.array(dets,dtype=np.float32), frame) if dets else []

        for t in tracks:
            x1,y1,x2,y2,tid = map(int, t[:5])

            # smoothing
            if tid not in prev_boxes:
                prev_boxes[tid] = [x1,y1,x2,y2]
            else:
                px1,py1,px2,py2 = prev_boxes[tid]
                x1 = int(alpha*px1+(1-alpha)*x1)
                y1 = int(alpha*py1+(1-alpha)*y1)
                x2 = int(alpha*px2+(1-alpha)*x2)
                y2 = int(alpha*py2+(1-alpha)*y2)
                prev_boxes[tid] = [x1,y1,x2,y2]

            key = tid

            # ReID trigger logic
            run_reid = False

            if key not in tid_to_pid:
                run_reid = True
            elif frame_index - tid_last_reid_frame.get(key, 0) >= REID_REFRESH_FRAMES:
                run_reid = True

            if run_reid:
                person_crop = frame[y1:y2, x1:x2].copy()
                if person_crop.size != 0 and not face_queue.full():
                    face_queue.put((person_crop, key, frame_index))

            pid = tid_to_pid.get(key, -1)

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,
                        f"TID {tid} PID {pid}",
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,(0,255,0),2)

        # Metrics
        second_process += 1
        latency = time.time() - start
        latency_second += latency
        max_latency = max(max_latency, latency)

        current_time = time.time()
        if current_time - last_stats_time >= 1.0:
            elapsed = current_time - last_stats_time

            print("\n📊 MINI-DUAL-PIPELINE METRICS")
            print(f"Capture FPS: {second_capture/elapsed:.2f}")
            print(f"Processing FPS: {second_process/elapsed:.2f}")
            print(f"Avg Latency: {(latency_second/second_process)*1000:.2f} ms")
            print(f"Max Latency: {max_latency*1000:.2f} ms")

            second_capture = 0
            second_process = 0
            latency_second = 0
            max_latency = 0
            last_stats_time = current_time

        cv2.imshow("Retail Optimized Pipeline", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# =========================================================
# START THREADS
# =========================================================
threading.Thread(target=decode_thread, daemon=True).start()
threading.Thread(target=inference_thread, daemon=True).start()
threading.Thread(target=face_worker, daemon=True).start()

while True:
    time.sleep(1)
