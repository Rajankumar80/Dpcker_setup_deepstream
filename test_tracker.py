import cv2
import numpy as np
import time
import threading
from queue import Queue
from boxmot import ByteTrack
from openvino import Core
import os

# Limit OpenVINO threads
os.environ["OMP_NUM_THREADS"] = "4"

# =========================================================
# CONFIG
# =========================================================
CAMERA_URL = "rtsp://10.64.36.14:554/rtsp/streaming?channel=01&subtype=1"

yolo_xml = r"C:\Users\TP-User\Desktop\store_analytics_proj\yolov8n_int8_openvino_model\yolov8n.xml"

FACE_DET_PATH = r"C:\Users\TP-User\Desktop\store_analytics_proj\face-detection-retail-0004.xml"
FACE_REID_PATH = r"C:\Users\TP-User\Desktop\store_analytics_proj\face-reidentification-retail-0095.xml"

FACE_THRESHOLD = 0.65
REID_REFRESH_FRAMES = 40

# =========================================================
# QUEUES
# =========================================================
frame_queue = Queue(maxsize=5)
face_queue = Queue(maxsize=50)

# =========================================================
# GLOBAL MEMORY
# =========================================================
pid_embeddings = {}
tid_to_pid = {}
tid_last_reid_frame = {}
next_pid = 1
pid_lock = threading.Lock()


# =========================================================
# OPENVINO INIT (FACE MODELS)
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

        if not frame_queue.full():
            frame_queue.put(frame)

        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            print(f"Camera FPS: {capture_counter / elapsed:.2f}")
            capture_counter = 0
            start_time = time.time()

# =========================================================
# THREAD 3: FACE WORKER
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
    frame_id = 0

    core = Core()
    model = core.read_model(yolo_xml)

    compiled_model = core.compile_model(
        model,
        "CPU",
        {
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_NUM_THREADS": "4"
        }
    )

    output_layer = compiled_model.output(0)
    input_layer = compiled_model.input(0)
    _, _, INPUT_H, INPUT_W = input_layer.shape

    tracker = ByteTrack(track_thresh=0.4, match_thresh=0.8, min_hits=2)

    prev_boxes = {}
    alpha = 0.8

    second_capture = 0
    second_process = 0
    latency_sum = 0
    max_latency = 0
    last_stats_time = time.time()

    # ---------------- Letterbox Function ----------------
    def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
        shape = img.shape[:2]  # current shape [h, w]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r

        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=color)

        return img, ratio, dw, dh
    # ----------------------------------------------------

    while True:

        if frame_queue.empty():
            continue

        start_time = time.time()
        frame = frame_queue.get()
        second_capture += 1


        orig_h, orig_w = frame.shape[:2]

        # ============== LETTERBOX PREPROCESS ==============
        img, ratio, dw, dh = letterbox(frame, (INPUT_H, INPUT_W))
        img = img.transpose(2, 0, 1)
        img = img[None].astype(np.float32) / 255.0

        # ============== INFERENCE =========================
       # ============== INFERENCE =========================
        result = compiled_model([img])[output_layer]

        pred = np.squeeze(result)      # (84, 8400)
        pred = pred.transpose(1, 0)    # (8400, 84)

        boxes = []
        scores = []

        for row in pred:

            cx, cy, w, h = row[:4]

            # ✅ YOLOv8 OpenVINO decode (NO multiplication)
            class_scores = row[4:]
            class_id = np.argmax(class_scores)
            conf = class_scores[class_id]

            if class_id != 0:  # person
                continue

            if conf < 0.25:
                continue

            # Convert to xyxy
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            # Undo letterbox
            x1 = (x1 - dw) / ratio
            y1 = (y1 - dh) / ratio
            x2 = (x2 - dw) / ratio
            y2 = (y2 - dh) / ratio

            x1 = int(max(0, min(orig_w, x1)))
            y1 = int(max(0, min(orig_h, y1)))
            x2 = int(max(0, min(orig_w, x2)))
            y2 = int(max(0, min(orig_h, y2)))

            boxes.append([x1, y1, x2, y2])
            scores.append(float(conf))

        dets = []

        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45)

            if len(indices) > 0:
                for i in indices.flatten():
                    x1, y1, x2, y2 = boxes[i]
                    dets.append([x1, y1, x2, y2, scores[i], 0])

        dets = np.array(dets, dtype=np.float32) if len(dets) > 0 else []

        tracks = tracker.update(dets, frame) if len(dets) > 0 else []

        for t in tracks:
            x1, y1, x2, y2, tid = map(int, t[:5])

            if tid not in prev_boxes:
                prev_boxes[tid] = [x1, y1, x2, y2]
            else:
                px1, py1, px2, py2 = prev_boxes[tid]
                x1 = int(alpha * px1 + (1 - alpha) * x1)
                y1 = int(alpha * py1 + (1 - alpha) * y1)
                x2 = int(alpha * px2 + (1 - alpha) * x2)
                y2 = int(alpha * py2 + (1 - alpha) * y2)
                prev_boxes[tid] = [x1, y1, x2, y2]

            frame_id += 1

            if tid not in tid_last_reid_frame or \
            frame_id - tid_last_reid_frame.get(tid, 0) > REID_REFRESH_FRAMES:

                person_crop = frame[y1:y2, x1:x2]

                if person_crop.size > 0 and not face_queue.full():
                    face_queue.put((person_crop.copy(), tid, frame_id))

            pid = tid_to_pid.get(tid, None)

            if pid:
                label = f"TID {tid} | PID {pid}"
            else:
                label = f"TID {tid}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

        # ============== METRICS ==========================
        second_process += 1
        latency = time.time() - start_time 
        latency_sum += latency
        max_latency = max(max_latency, latency)

        current_time = time.time()
        if current_time - last_stats_time >= 1.0:
            elapsed = current_time - last_stats_time

            print("\n📊 MINI-DUAL-PIPELINE METRICS")
            print(f"Capture FPS: {second_capture / elapsed:.2f}")
            print(f"Processing FPS: {second_process / elapsed:.2f}")
            print(f"Avg Latency: {(latency_sum / second_process) * 1000:.2f} ms")
            print(f"Max Latency: {max_latency * 1000:.2f} ms")

            second_capture = 0
            second_process = 0
            latency_sum = 0
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
