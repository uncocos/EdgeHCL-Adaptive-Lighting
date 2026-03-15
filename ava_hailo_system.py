import sys
import os
import cv2
import mediapipe as mp
import numpy as np
import time
import math
import json
import paho.mqtt.client as mqtt
from collections import deque
import board
import busio
from adafruit_pca9685 import PCA9685
import warnings

# Suppress specific hardware warnings to maintain clean terminal output
warnings.filterwarnings('ignore', category=UserWarning, module='sct')

# Configure module paths for Hailo-8L NPU inference integration
current_dir = os.path.dirname(os.path.abspath(__file__))
if not current_dir: current_dir = os.getcwd()
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '../blaze_common')))
from blazedetector import BlazeDetector
from hailo_inference import HailoInference

# ======================== 1. Hardware Initialization: PCA9685 & LED ========================
try:
    # Initialize I2C communication for PWM controller
    i2c = busio.I2C(board.SCL, board.SDA)
    pca = PCA9685(i2c)
    pca.frequency = 1000 # Set high frequency for flicker-free lighting
    print("PCA9685 hardware driver initialized successfully")
except Exception as e:
    print(f"Failed to initialize PCA9685: {e}")
    exit()

# Define PWM channel indices for RGB LED control
RED_CH, GREEN_CH, BLUE_CH = 0, 1, 2

def set_led_pwm(channel, value):
    # Map 0.0-1.0 float to 16-bit duty cycle (0-65535)
    duty = int(value * 65535)
    pca.channels[channel].duty_cycle = max(0, min(65535, duty))

# Define RGB color coefficients for different Correlated Color Temperatures (CCT)
COLOR_WARM = (1.0, 0.5, 0.1)
COLOR_COOL = (0.7, 0.9, 1.0)

SYS_MODES = ["MANUAL", "WARM", "COOL", "AI_MODE"]
current_sys_mode = 0
is_power_on = True
adjust_param = "brightness"
manual_brightness, manual_temp = 0.8, 0.5
ai_brightness, ai_temp, current_lux = 0.8, 0.5, 0.0

# ======================== 2. Auxiliary Functions & Visualization ========================
def calculate_distance(p1, p2): 
    # Euclidean distance calculation for facial landmark analysis
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def compute_EAR(eye_points):
    # Eye Aspect Ratio calculation to determine blink states and duration
    p1, p2, p3, p4, p5, p6 = eye_points
    return (calculate_distance(p2, p6) + calculate_distance(p3, p5)) / (2.0 * calculate_distance(p1, p4) + 1e-6)

def compute_MAR(mouth_points): 
    # Mouth Aspect Ratio calculation for yawning detection
    return calculate_distance(mouth_points[2], mouth_points[3]) / (calculate_distance(mouth_points[0], mouth_points[1]) + 1e-6)

def compute_brow_furrow_ratio(brow_p1, brow_p2, eye_p1, eye_p2): 
    # Cognitive load estimation via eyebrow furrowing ratio
    return calculate_distance(brow_p1, brow_p2) / (calculate_distance(eye_p1, eye_p2) + 1e-6)

def get_gaze_angle(landmarks, frame_shape):
    # Estimate 3D head pose vector based on nasal and mandibular anchor points
    h, w = frame_shape
    face_3d = np.array([
        (landmarks[1].x * w, landmarks[1].y * h, landmarks[1].z * w), # Nose tip
        (landmarks[9].x * w, landmarks[9].y * h, landmarks[9].z * w)  # Chin center
    ], dtype="double")
    return face_3d[0] - face_3d[1]

def calculate_angle_between_vectors(v1, v2):
    # Calculate angular deviation for gaze tracking analysis
    unit_v1 = v1 / (np.linalg.norm(v1) + 1e-6)
    unit_v2 = v2 / (np.linalg.norm(v2) + 1e-6)
    dot_product = np.dot(unit_v1, unit_v2)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return np.degrees(angle)

def put_text_on_frame(frame, text, y_pos, color=(0, 255, 0)):
    # Standardized UI text rendering
    cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return y_pos + 30

class LiveGraph:
    # Real-time data visualization tool for EAR and AVA metrics
    def __init__(self, w, h, max_len=100, min_val=0, max_val=100, color=(0, 255, 0), title="Graph"):
        self.w, self.h = w, h
        self.data = deque([min_val]*max_len, maxlen=max_len)
        self.max_len, self.min_val, self.max_val = max_len, min_val, max_val
        self.color, self.title = color, title

    def update(self, frame, new_value, x_offset, y_offset):
        self.data.append(new_value)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_offset, y_offset), (x_offset + self.w, y_offset + self.h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, f"{self.title}: {new_value:.2f}", (x_offset + 5, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if len(self.data) > 1:
            pts = []
            for i, val in enumerate(self.data):
                x = x_offset + int(i * (self.w / self.max_len))
                norm_val = max(0, min(1, (val - self.min_val) / (self.max_val - self.min_val + 1e-6)))
                y = y_offset + self.h - int(norm_val * (self.h - 10)) - 5
                pts.append((x, y))
            for i in range(1, len(pts)): cv2.line(frame, pts[i-1], pts[i], self.color, self.color, 2)

class GlobalLandmark:
    # Coordinate mapper to re-project localized ROI landmarks back to global screen space
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

# ======================== 3. AVA Algorithm Logic ========================
def compute_Pareto_freq(ibis, fs=10.0):
    # Calculate Pareto Frequency from Inter-Blink Intervals (IBI) via FFT
    if len(ibis) < 3: return None
    total_time = np.sum(ibis)
    if total_time < 0.5: return None
    
    num_samples = max(10, int(total_time * fs))
    t = np.linspace(0, total_time, num_samples)
    blink_cumulative_times = np.cumsum(np.insert(ibis, 0, 0))
    ibi_step_func = np.interp(t, blink_cumulative_times[:-1], ibis, right=ibis[-1])
    
    x = ibi_step_func - np.mean(ibi_step_func)
    N = len(x)
    psd = np.abs(np.fft.rfft(x * np.hanning(N)))**2
    freqs = np.fft.rfftfreq(N, d=1.0/fs)
    
    total_power = np.sum(psd)
    if total_power == 0: return None
    
    cumulative_power = np.cumsum(psd)
    idx = np.where(cumulative_power >= 0.8 * total_power)[0][0]
    return freqs[idx]

def compute_AVA(blink_times, nad_percentage, omega_r=0.08):
    # Unified Attention Visual Analyzer score calculation
    if len(blink_times) < 3: return 50.0 
    ibis = np.diff(list(blink_times))
    omega = compute_Pareto_freq(ibis)
    if omega is None or omega == 0: return 50.0
    phi_base = 100 * (omega_r / omega)
    phi_final = ((100.0 - nad_percentage) / 100.0) * phi_base
    return max(0.0, min(100.0, phi_final))

# ======================== 4. MQTT Protocol Implementation ========================
def on_connect(client, userdata, flags, rc):
    # Subscribe to relevant topics upon successful connection
    client.subscribe("light/control")
    client.subscribe("light/monitor")

def on_message(client, userdata, msg):
    # Dynamic parameter updates via MQTT messages
    global current_sys_mode, is_power_on, adjust_param
    global manual_brightness, manual_temp, current_lux
    try:
        data = json.loads(msg.payload.decode())
        if msg.topic == "light/monitor":
            current_lux = data.get("value", 0.0)
        elif msg.topic == "light/control":
            action = data.get("action")
            if action == "power_toggle": is_power_on = not is_power_on
            elif action == "switch_mode": current_sys_mode = (current_sys_mode + 1) % 4
            elif action == "toggle_param": 
                if current_sys_mode == 0: adjust_param = "temp" if adjust_param == "brightness" else "brightness"
            elif action == "adjust":
                step = data.get("step", 0) / 100.0
                if current_sys_mode == 0:
                    if adjust_param == "brightness": manual_brightness = max(0.1, min(1.0, manual_brightness + step))
                    else: manual_temp = max(0.0, min(1.0, manual_temp + step))
                elif current_sys_mode in [1, 2]: manual_brightness = max(0.1, min(1.0, manual_brightness + step))
    except: pass

mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.connect("localhost", 1883, 60)
mqtt_client.loop_start()

# ======================== 5. Initialization: NPU & MediaPipe ========================
print("Initializing Hailo-8L NPU...")
hailo_inference = HailoInference()
blaze_detector = BlazeDetector("blazeface", hailo_infer=hailo_inference)
blaze_detector.load_model('models/face_detection_short_range.hef')

print("Initializing MediaPipe FaceMesh (CPU)...")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
cap = cv2.VideoCapture(0)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
MOUTH = [61, 291, 13, 14]
FACE_LMKS = set(LEFT_EYE + RIGHT_EYE + MOUTH + [107, 336, 130, 359, 1, 9])

DISTRACTION_ANGLE_THRESH = 20.0 # Degrees

start_time = time.time()
calibration_complete = False
calib_gaze_vectors = []
gaze_vector_ref = None
blink_times_window = deque()
head_pose_window = deque(maxlen=100)
ava_scores_window = deque(maxlen=60)

stable_ava, EAR, MAR, brow_ratio, nad = 50.0, 0.0, 0.0, 0.0, 0.0
blink_count, closed_frames, yawn_count, yawning_frames = 0, 0, 0, 0
is_looking_away, prev_frame_time = False, 0
current_attention_state = "NEUTRAL"
last_state_change_time = 0

graph_ear = LiveGraph(250, 100, min_val=0.0, max_val=0.4, color=(255, 255, 0), title="EAR")
graph_ava = LiveGraph(250, 100, min_val=0, max_val=100, color=(0, 165, 255), title="AVA Score")

# Finite State Machine variables for heterogeneous pipeline control
tracking_active = False
roi_box = None 

# ======================== 6. Main Execution Loop ========================
while cap.isOpened():
    success, frame = cap.read()
    if not success: continue

    curr_t = time.time()
    fps = 1 / (curr_t - prev_frame_time) if prev_frame_time > 0 else 0
    prev_frame_time = curr_t
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    elapsed = curr_t - start_time
    is_looking_away = False
    
    # ======================== Heterogeneous Pipeline: NPU-CPU Handoff ========================
    lms = None # Landmark buffer
    
    if not tracking_active:
        # State: Macro-Search via NPU (INT8 quantized model)
        cv2.putText(frame, "STATUS: NPU MACRO-SEARCH", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        img_npu, scale, pad = blaze_detector.resize_pad(frame)
        normalized_detections = blaze_detector.predict_on_image(img_npu)
        
        if isinstance(normalized_detections, np.ndarray) and len(normalized_detections) > 0:
            detections = blaze_detector.denormalize_detections(normalized_detections, scale, pad)
            det = detections[0]
            ymin, xmin, ymax, xmax = det[0], det[1], det[2], det[3]
            
            bw_npu, bh_npu = xmax - xmin, ymax - ymin
            cx, cy = xmin + bw_npu/2, ymin + bh_npu/2
            crop_size = max(bw_npu, bh_npu) * 1.4 # Expand ROI for better context
            
            x1 = max(0, int(cx - crop_size/2))
            y1 = max(0, int(cy - crop_size/2))
            x2 = min(w, int(cx + crop_size/2))
            y2 = min(h, int(cy + crop_size/2))
            
            roi_box = (x1, y1, x2, y2)
            tracking_active = True
            
    if tracking_active and roi_box is not None:
        # State: Micro-Tracking via CPU (localized ROI for efficiency)
        cv2.putText(frame, "STATUS: CPU MICRO-TRACKING", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        x1, y1, x2, y2 = roi_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # Render ROI boundary
        
        roi_frame = frame[y1:y2, x1:x2]
        roi_h, roi_w = roi_frame.shape[:2]
        
        if roi_h > 0 and roi_w > 0:
            results = face_mesh.process(cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                # Reverse mapping: localized ROI landmarks to global coordinate space
                global_landmarks = []
                bbox_bounds = [w, h, 0, 0] # Initial min_x, min_y, max_x, max_y
                
                for lm in face_landmarks.landmark:
                    gx = (lm.x * roi_w + x1) / w
                    gy = (lm.y * roi_h + y1) / h
                    gz = lm.z * (roi_w / w) # Scale normalization for Z-axis
                    global_landmarks.append(GlobalLandmark(gx, gy, gz))
                    
                    abs_x, abs_y = int(gx * w), int(gy * h)
                    if abs_x < bbox_bounds[0]: bbox_bounds[0] = abs_x
                    if abs_x > bbox_bounds[2]: bbox_bounds[2] = abs_x
                    if abs_y < bbox_bounds[1]: bbox_bounds[1] = abs_y
                    if abs_y > bbox_bounds[3]: bbox_bounds[3] = abs_y
                
                lms = global_landmarks
                
                # Predictive ROI update for the subsequent frame
                min_x, min_y, max_x, max_y = bbox_bounds
                bw_new, bh_new = max_x - min_x, max_y - min_y
                cx_new, cy_new = min_x + bw_new/2, min_y + bh_new/2
                crop_size_new = max(bw_new, bh_new) * 1.5
                
                nx1 = max(0, int(cx_new - crop_size_new/2))
                ny1 = max(0, int(cy_new - crop_size_new/2))
                nx2 = min(w, int(cx_new + crop_size_new/2))
                ny2 = min(h, int(cy_new + crop_size_new/2))
                roi_box = (nx1, ny1, nx2, ny2)
            else:
                tracking_active = False # Fallback to NPU search if tracking fails
        else:
            tracking_active = False

    # ======================== Data Processing & State Inference ========================
    if lms is not None:
        def p(i): return (int(lms[i].x * w), int(lms[i].y * h))
        
        for idx in FACE_LMKS:
            cv2.circle(frame, p(idx), 2, (0, 255, 255), -1)

        # Quantitative ocular and oral feature analysis
        EAR = (compute_EAR([p(i) for i in LEFT_EYE]) + compute_EAR([p(i) for i in RIGHT_EYE])) / 2.0
        MAR = compute_MAR([p(i) for i in MOUTH])
        brow_ratio = compute_brow_furrow_ratio(p(107), p(336), p(130), p(359))
        
        current_gaze_vector = get_gaze_angle(lms, (h, w))

        if not calibration_complete:
            # System calibration phase for head pose baseline
            if elapsed <= 5: calib_gaze_vectors.append(current_gaze_vector)
            else:
                gaze_vector_ref = np.mean(calib_gaze_vectors, axis=0)
                calibration_complete, start_time = True, time.time()
        else:
            # Blink detection state machine
            if EAR < 0.20: closed_frames += 1
            else:
                if closed_frames >= 2: 
                    blink_count += 1
                    blink_times_window.append(elapsed)
                closed_frames = 0
            
            # Yawn detection logic
            is_yawning = False
            if MAR > 0.55: yawning_frames += 1
            else:
                if yawning_frames >= 5: 
                    yawn_count += 1
                yawning_frames = 0
            
            if yawning_frames > 2: is_yawning = True

            # Gaze deviation and distraction detection
            head_angle_dev = calculate_angle_between_vectors(current_gaze_vector, gaze_vector_ref)
            if head_angle_dev > DISTRACTION_ANGLE_THRESH or is_yawning: 
                is_looking_away = True
            
            head_pose_window.append(1 if is_looking_away else 0)
            
            # Maintenance of the temporal data window
            while blink_times_window and blink_times_window[0] < elapsed - 60: 
                blink_times_window.popleft()
            
            # Final attention state classification
            nad = (sum(head_pose_window)/len(head_pose_window)) * 100 if len(head_pose_window) > 0 else 0
            ava = compute_AVA(blink_times_window, nad)
            ava_scores_window.append(ava)
            stable_ava = np.mean(ava_scores_window)

            if curr_t - last_state_change_time > 10:
                if stable_ava > 75: current_attention_state = "FOCUSED"
                elif stable_ava < 40: current_attention_state = "DISTRACTED"
                else: current_attention_state = "NEUTRAL"
                last_state_change_time = curr_t

            graph_ear.update(frame, EAR, w - 270, h - 230)
            graph_ava.update(frame, stable_ava, w - 270, h - 110)

    # ======================== 7. Adaptive HCL Control: Decision Engine ========================
    if not is_power_on:
        set_led_pwm(RED_CH, 0); set_led_pwm(GREEN_CH, 0); set_led_pwm(BLUE_CH, 0)
    else:
        if current_sys_mode == 0: t, b = manual_temp, manual_brightness
        elif current_sys_mode == 1: t, b = 0.0, manual_brightness
        elif current_sys_mode == 2: t, b = 1.0, manual_brightness
        else:
            # Autonomous AI mapping: Physiological state to lighting parameters
            target_lux = 600 if current_attention_state == "FOCUSED" else (250 if current_attention_state == "DISTRACTED" else 400)
            target_t = 1.0 if current_attention_state == "FOCUSED" else (0.0 if current_attention_state == "DISTRACTED" else 0.5)
            ai_temp += (target_t - ai_temp) * 0.02 # Temporal smoothing
            if current_lux > 0:
                err = target_lux - current_lux
                if abs(err) > 30: ai_brightness = max(0.1, min(1.0, ai_brightness + (0.005 if err > 0 else -0.005)))
            t, b = ai_temp, ai_brightness

        # Final PWM output generation for the LED hardware
        r = (COLOR_WARM[0] + (COLOR_COOL[0] - COLOR_WARM[0]) * t) * b
        g = (COLOR_WARM[1] + (COLOR_COOL[1] - COLOR_WARM[1]) * t) * b
        bl = (COLOR_WARM[2] + (COLOR_COOL[2] - COLOR_WARM[2]) * t) * b
        set_led_pwm(RED_CH, r); set_led_pwm(GREEN_CH, g); set_led_pwm(BLUE_CH, bl)

    # ======================== 8. User Interface Rendering ========================
    if not calibration_complete:
        cv2.putText(frame, f"CALIBRATING: {max(0, 5-elapsed):.1f}s", (int(w/2-150), int(h/2)), 1, 2, (0, 0, 255), 2)
    else:
        bpm = len(blink_times_window) 
        y = 30
        y = put_text_on_frame(frame, f"SYS: {SYS_MODES[current_sys_mode]} ({'BRIGHT' if adjust_param=='brightness' else 'TEMP'})", y, (255, 255, 255))
        y = put_text_on_frame(frame, f"EAR: {EAR:.3f}", y)
        y = put_text_on_frame(frame, f"BLINK/MIN: {bpm}", y, (255, 255, 0))
        y = put_text_on_frame(frame, f"YAWNS: {yawn_count}", y, (0, 255, 255))
        y = put_text_on_frame(frame, f"NAD: {nad:.1f}%", y, (0, 165, 255))
        y = put_text_on_frame(frame, f"COGNITIVE: {brow_ratio:.3f}", y, (255, 0, 255))
        
        y += 10
        y = put_text_on_frame(frame, f"ENV LUX: {current_lux:.1f}", y, (200, 255, 200))
        if current_sys_mode == 3:
            y = put_text_on_frame(frame, f"AI TEMP: {ai_temp:.2f}", y, (150, 200, 255))
            y = put_text_on_frame(frame, f"AI BRIGHT: {ai_brightness:.2f}", y, (150, 200, 255))
        
        y += 10
        y = put_text_on_frame(frame, f"AVA SCORE: {stable_ava:.1f}", y, (0, 255, 0))
        s_color = (0, 0, 255) if current_attention_state == "DISTRACTED" else ((0, 255, 255) if current_attention_state == "NEUTRAL" else (0, 255, 0))
        y = put_text_on_frame(frame, f"STATE: {current_attention_state}", y, s_color)
        
        if is_looking_away: 
            cv2.putText(frame, "WARNING: DISTRACTED", (w-250, 30), 1, 1.5, (0, 0, 255), 2)

    cv2.putText(frame, f"FPS: {int(fps)}", (w-120, h-20), 1, 1.2, (0, 255, 0), 2)
    cv2.imshow('EdgeHCL: Adaptive Lighting System', frame)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

# Resource cleanup
cap.release()
cv2.destroyAllWindows()
mqtt_client.loop_stop()
set_led_pwm(RED_CH, 0); set_led_pwm(GREEN_CH, 0); set_led_pwm(BLUE_CH, 0)