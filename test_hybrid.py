import sys
import os
import math
import cv2
import time
import numpy as np
import mediapipe as mp

# Configure library paths for Hailo-8L integration
current_dir = os.path.dirname(os.path.abspath(__file__))
if not current_dir:
    current_dir = os.getcwd()
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '../blaze_common')))

from blazedetector import BlazeDetector
from hailo_inference import HailoInference

# Define geometric calculation functions
def calculate_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def eye_aspect_ratio(eye_pts):
    v1 = calculate_distance(eye_pts[1], eye_pts[5])
    v2 = calculate_distance(eye_pts[2], eye_pts[4])
    h = calculate_distance(eye_pts[0], eye_pts[3])
    if h == 0: return 0
    return (v1 + v2) / (2.0 * h)

def mouth_aspect_ratio(mouth_pts):
    v = calculate_distance(mouth_pts[1], mouth_pts[3])
    h = calculate_distance(mouth_pts[0], mouth_pts[2])
    if h == 0: return 0
    return v / h

EAR_THRESH = 0.20 
MAR_THRESH = 0.50 

print("[INFO] Initializing NPU for Macro-Detection...")
hailo_inference = HailoInference()
blaze_detector = BlazeDetector("blazeface", hailo_infer=hailo_inference)
blaze_detector.load_model('models/face_detection_short_range.hef')

print("[INFO] Initializing CPU for Micro-Landmarking...")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Standard Camera Configuration
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

tracking_active = False
roi_box = None 

# Benchmarking variables for academic performance analysis
frame_count = 0
MEASURE_FRAMES = 500  
total_inference_time = 0.0 # Cumulative inference latency

print("[INFO] Heterogeneous Benchmark Started...")

while cap.isOpened():
    ret, frame = cap.read() # I/O time excluded from latency measurement
    if not ret: break
    frame = cv2.flip(frame, 1)
    h_img, w_img = frame.shape[:2]
    
    frame_count += 1
    if frame_count == 50:
        print("\n[INFO] Warm-up complete. Starting latency measurement...")
        
    ear_val, mar_val = 0.0, 0.0
    status_text, status_color = "Awake", (0, 255, 0)
    
    # ⏱️ Start Latency Timer: Capture raw AI processing time
    infer_start = time.time()
    
    if not tracking_active:
        # NPU Stage: Global search
        img_npu, scale, pad = blaze_detector.resize_pad(frame)
        normalized_detections = blaze_detector.predict_on_image(img_npu)
        
        if isinstance(normalized_detections, np.ndarray) and len(normalized_detections) > 0:
            detections = blaze_detector.denormalize_detections(normalized_detections, scale, pad)
            det = detections[0]
            ymin, xmin, ymax, xmax = det[0], det[1], det[2], det[3]
            
            bw, bh = xmax - xmin, ymax - ymin
            cx, cy = xmin + bw/2, ymin + bh/2
            crop_size = max(bw, bh) * 1.4
            
            x1 = max(0, int(cx - crop_size/2))
            y1 = max(0, int(cy - crop_size/2))
            x2 = min(w_img, int(cx + crop_size/2))
            y2 = min(h_img, int(cy + crop_size/2))
            
            roi_box = (x1, y1, x2, y2)
            tracking_active = True 
            
    if tracking_active and roi_box is not None:
        # CPU Stage: Localized mesh tracking
        x1, y1, x2, y2 = roi_box
        roi_frame = frame[y1:y2, x1:x2]
        roi_h, roi_w = roi_frame.shape[:2]

        if roi_h > 0 and roi_w > 0:
            roi_rgb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(roi_rgb)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                bbox_bounds = [w_img, h_img, 0, 0]

                def get_pt(idx):
                    pt = face_landmarks.landmark[idx]
                    global_x = int(pt.x * roi_w) + x1
                    global_y = int(pt.y * roi_h) + y1
                    
                    if global_x < bbox_bounds[0]: bbox_bounds[0] = global_x
                    if global_x > bbox_bounds[2]: bbox_bounds[2] = global_x
                    if global_y < bbox_bounds[1]: bbox_bounds[1] = global_y
                    if global_y > bbox_bounds[3]: bbox_bounds[3] = global_y
                    return (global_x, global_y)

                # Landmark extraction for state inference
                left_eye = [get_pt(33), get_pt(160), get_pt(158), get_pt(133), get_pt(153), get_pt(144)]
                right_eye = [get_pt(362), get_pt(385), get_pt(387), get_pt(263), get_pt(373), get_pt(380)]
                mouth = [get_pt(78), get_pt(13), get_pt(308), get_pt(14)]
                
                for idx in [10, 152, 234, 454]: get_pt(idx)

                ear_val = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
                mar_val = mouth_aspect_ratio(mouth)

                if ear_val < EAR_THRESH:
                    status_text, status_color = "DROWSY", (0, 0, 255)
                elif mar_val > MAR_THRESH:
                    status_text, status_color = "YAWNING", (0, 165, 255)

                # Dynamically update ROI for next frame
                min_x, min_y, max_x, max_y = bbox_bounds
                bw, bh = max_x - min_x, max_y - min_y
                cx, cy = min_x + bw/2, min_y + bh/2
                crop_size = max(bw, bh) * 1.5 
                
                nx1 = max(0, int(cx - crop_size/2))
                ny1 = max(0, int(cy - crop_size/2))
                nx2 = min(w_img, int(cx + crop_size/2))
                ny2 = min(h_img, int(cy + crop_size/2))
                roi_box = (nx1, ny1, nx2, ny2)
            else:
                tracking_active = False 

    # ⏱️ End Latency Timer
    infer_end = time.time()
    
    # Statistical logging
    if 50 < frame_count <= 50 + MEASURE_FRAMES:
        total_inference_time += (infer_end - infer_start)
        
        if frame_count == 50 + MEASURE_FRAMES:
            avg_latency = (total_inference_time / MEASURE_FRAMES) * 1000 # Convert to ms
            potential_fps = 1000.0 / avg_latency if avg_latency > 0 else 0
            
            print("\n" + "="*45)
            print(f"Heterogeneous Latency Benchmark Final Results:")
            print(f"Total Measured Frames: {MEASURE_FRAMES}")
            print(f"Average Inference Latency: {avg_latency:.2f} ms")
            print(f"Potential Throughput (FPS): {potential_fps:.2f}")
            print("="*45 + "\n")
            break

    # Real-time monitoring UI
    if tracking_active and roi_box is not None:
        cv2.putText(frame, "STATE: CPU TRACKING", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "STATE: NPU SEARCHING", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(frame, f"EAR: {ear_val:.2f} | STATUS: {status_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow("Heterogeneous Inference Benchmark", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()