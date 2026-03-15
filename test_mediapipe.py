import cv2
import time
import math
import mediapipe as mp

# ======================== 1. Feature Extraction Algorithms ========================
def calculate_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def eye_aspect_ratio(eye_pts):
    # Standard formula for quantifying ocular aperture
    v1 = calculate_distance(eye_pts[1], eye_pts[5])
    v2 = calculate_distance(eye_pts[2], eye_pts[4])
    h = calculate_distance(eye_pts[0], eye_pts[3])
    if h == 0: return 0
    return (v1 + v2) / (2.0 * h)

def mouth_aspect_ratio(mouth_pts):
    # Standard formula for oral aperture quantification
    v = calculate_distance(mouth_pts[1], mouth_pts[3])
    h = calculate_distance(mouth_pts[0], mouth_pts[2])
    if h == 0: return 0
    return v / h

EAR_THRESH = 0.20 
MAR_THRESH = 0.50 

# ======================== 2. System Initialization (Pure CPU) ========================
print("[INFO] Initializing MediaPipe CPU Pipeline (Baseline)...")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, 
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Standardize Video Pipeline
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("[INFO] Baseline Latency Benchmark Started...")

# Benchmarking telemetry
frame_count = 0
MEASURE_FRAMES = 500  
total_inference_time = 0.0

while cap.isOpened():
    ret, frame = cap.read() # I/O latency is excluded from inference benchmarking
    if not ret: break
    frame = cv2.flip(frame, 1)
    h_img, w_img = frame.shape[:2]
    
    frame_count += 1
    if frame_count == 50:
        print("\n[INFO] Warm-up complete. Commencing CPU latency measurement...")
        
    ear_val, mar_val = 0.0, 0.0
    status_text, status_color = "Awake", (0, 255, 0)
    
    # ⏱️ Start Latency Timer: Capturing raw computational cost
    infer_start = time.time()
    
    # Core Processing: Global CPU FaceMesh inference
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        
        def get_pt(idx):
            pt = face_landmarks.landmark[idx]
            return (int(pt.x * w_img), int(pt.y * h_img))

        # Physiological landmark extraction
        left_eye = [get_pt(33), get_pt(160), get_pt(158), get_pt(133), get_pt(153), get_pt(144)]
        right_eye = [get_pt(362), get_pt(385), get_pt(387), get_pt(263), get_pt(373), get_pt(380)]
        mouth = [get_pt(78), get_pt(13), get_pt(308), get_pt(14)]

        ear_val = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        mar_val = mouth_aspect_ratio(mouth)

        if ear_val < EAR_THRESH:
            status_text, status_color = "DROWSY", (0, 0, 255)
        elif mar_val > MAR_THRESH:
            status_text, status_color = "YAWNING", (0, 165, 255)

    # ⏱️ End Latency Timer
    infer_end = time.time()
    
    # Performance telemetry collection
    if 50 < frame_count <= 50 + MEASURE_FRAMES:
        total_inference_time += (infer_end - infer_start)
        
        if frame_count == 50 + MEASURE_FRAMES:
            avg_latency = (total_inference_time / MEASURE_FRAMES) * 1000 
            potential_fps = 1000.0 / avg_latency if avg_latency > 0 else 0
            
            print("\n" + "="*45)
            print(f"Standard CPU Latency Benchmark Final Results:")
            print(f"Total Measured Frames: {MEASURE_FRAMES}")
            print(f"Average Inference Latency: {avg_latency:.2f} ms")
            print(f"Potential Throughput (FPS): {potential_fps:.2f}")
            print("="*45 + "\n")
            break

    # Visualization
    if results.multi_face_landmarks:
        for pt in left_eye + right_eye + mouth:
            cv2.circle(frame, pt, 2, (0, 255, 0), -1)

    cv2.putText(frame, f"CPU LATENCY BENCHMARK: {status_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow("Pure CPU Baseline Benchmark", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()