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

# Suppress non-critical warnings for stable edge deployment
warnings.filterwarnings('ignore', category=UserWarning, module='sct')

# ======================== 1. Hardware Abstraction Layer: PCA9685 ========================
try:
    # Initialize I2C bus and PWM controller for hardware-level LED manipulation
    i2c = busio.I2C(board.SCL, board.SDA)
    pca = PCA9685(i2c)
    pca.frequency = 1000 
    print("PCA9685 hardware interface initialized successfully")
except Exception as e:
    print(f"Hardware Initialization Error: {e}")
    exit()

# Channel mapping for RGB LED strip
RED_CH, GREEN_CH, BLUE_CH = 0, 1, 2

def set_led_pwm(channel, value):
    # Convert 0.0-1.0 intensity to 16-bit PWM duty cycle
    duty = int(value * 65535)
    pca.channels[channel].duty_cycle = max(0, min(65535, duty))

# Predefined color profiles for Human-Centric Lighting (HCL)
COLOR_WARM = (1.0, 0.5, 0.1)
COLOR_COOL = (0.7, 0.9, 1.0)

SYS_MODES = ["MANUAL", "WARM", "COOL", "AI_MODE"]
current_sys_mode = 0
is_power_on = True
adjust_param = "brightness"
manual_brightness, manual_temp = 0.8, 0.5
ai_brightness, ai_temp, current_lux = 0.8, 0.5, 0.0

# ======================== 2. Signal Processing & Feature Extraction ========================
def calculate_distance(p1, p2): 
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def compute_EAR(eye_points):
    # Quantify eye opening degree via vertical/horizontal distance ratios
    p1, p2, p3, p4, p5, p6 = eye_points
    return (calculate_distance(p2, p6) + calculate_distance(p3, p5)) / (2.0 * calculate_distance(p1, p4) + 1e-6)

def compute_MAR(mouth_points): 
    # Quantify mouth opening degree for yawning event classification
    return calculate_distance(mouth_points[2], mouth_points[3]) / (calculate_distance(mouth_points[0], mouth_points[1]) + 1e-6)

def compute_brow_furrow_ratio(brow_p1, brow_p2, eye_p1, eye_p2): 
    # Calculate brow contraction ratio to estimate cognitive load
    return calculate_distance(brow_p1, brow_p2) / (calculate_distance(eye_p1, eye_p2) + 1e-6)

def get_gaze_angle(landmarks, frame_shape):
    # Estimate 3D head orientation vector relative to the camera plane
    h, w = frame_shape
    face_3d = np.array([
        (landmarks[1].x * w, landmarks[1].y * h, landmarks[1].z * w), # Nasal anchor
        (landmarks[9].x * w, landmarks[9].y * h, landmarks[9].z * w)  # Mandibular anchor
    ], dtype="double")
    return face_3d[0] - face_3d[1]

def calculate_angle_between_vectors(v1, v2):
    # Determine angular deviation for gaze distraction detection
    unit_v1 = v1 / (np.linalg.norm(v1) + 1e-6)
    unit_v2 = v2 / (np.linalg.norm(v2) + 1e-6)
    dot_product = np.dot(unit_v1, unit_v2)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return np.degrees(angle)

def put_text_on_frame(frame, text, y_pos, color=(0, 255, 0)):
    cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return y_pos + 30

class LiveGraph:
    # On-screen temporal data visualization for EAR and AVA metrics
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
            for i in range(1, len(pts)): cv2.line(frame, pts[i-1], pts[i], self.color, 2)

# ======================== 3. AVA Quantification Engine ========================
def compute_Pareto_freq(ibis, fs=10.0):
    # Frequency domain analysis of Inter-Blink Intervals (IBI)
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
    # Integrated Attention Visual Analyzer (AVA) score generation
    if len(blink_times) < 3: return 50.0 
    ibis = np.diff(list(blink_times))
    omega = compute_Pareto_freq(ibis)
    if omega is None or omega == 0: return 50.0
    phi_base = 100 * (omega_r / omega)
    phi_final = ((100.0 - nad_percentage) / 100.0) * phi_base
    return max(0.0, min(100.0, phi_final))

# ======================== 4. MQTT Connectivity Framework ========================
def on_connect(client, userdata, flags, rc):
    client.subscribe("light/control")
    client.subscribe("light/monitor")

def on_message(client, userdata, msg):
    # Asynchronous control logic for remote IoT interaction
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

# ======================== 5. MediaPipe Backend Initialization ========================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
cap = cv2.VideoCapture(0)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
MOUTH = [61, 291, 13, 14]
FACE_LMKS = set(LEFT_EYE + RIGHT_EYE + MOUTH + [107, 336, 130, 359, 1, 9])

DISTRACTION_ANGLE_THRESH = 20.0 

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

# ======================== 6. Main Processing Loop ========================
while cap.isOpened():
    success, frame = cap.read()
    if not success: continue

    curr_t = time.time()
    fps = 1 / (curr_t - prev_frame_time) if prev_frame_time > 0 else 0
    prev_frame_time = curr_t
    
    frame = cv2.flip(frame, 1)
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    h, w = frame.shape[:2]
    elapsed = curr_t - start_time
    is_looking_away = False

    if results.multi_face_landmarks:
        lms = results.multi_face_landmarks[0].landmark
        def p(i): return (int(lms[i].x * w), int(lms[i].y * h))
        
        for idx in FACE_LMKS:
            cv2.circle(frame, p(idx), 2, (0, 255, 255), -1)

        # Facial feature computation
        EAR = (compute_EAR([p(i) for i in LEFT_EYE]) + compute_EAR([p(i) for i in RIGHT_EYE])) / 2.0
        MAR = compute_MAR([p(i) for i in MOUTH])
        brow_ratio = compute_brow_furrow_ratio(p(107), p(336), p(130), p(359))
        
        current_gaze_vector = get_gaze_angle(lms, (h, w))

        if not calibration_complete:
            # Automatic head pose baseline calibration
            if elapsed <= 5: calib_gaze_vectors.append(current_gaze_vector)
            else:
                gaze_vector_ref = np.mean(calib_gaze_vectors, axis=0) 
                calibration_complete, start_time = True, time.time()
        else:
            # Blink state detection logic
            if EAR < 0.20: closed_frames += 1
            else:
                if closed_frames >= 2: 
                    blink_count += 1
                    blink_times_window.append(elapsed)
                closed_frames = 0
            
            # Drowsiness detection logic (yawn detection)
            is_yawning = False
            if MAR > 0.55: yawning_frames += 1
            else:
                if yawning_frames >= 5: 
                    yawn_count += 1
                yawning_frames = 0
            
            if yawning_frames > 2: is_yawning = True

            # Angular deviation analysis
            head_angle_dev = calculate_angle_between_vectors(current_gaze_vector, gaze_vector_ref)
            if head_angle_dev > DISTRACTION_ANGLE_THRESH or is_yawning: 
                is_looking_away = True
            
            head_pose_window.append(1 if is_looking_away else 0)
            
            while blink_times_window and blink_times_window[0] < elapsed - 60: 
                blink_times_window.popleft()
            
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

    # ======================== 7. HCL Decision Logic & PWM Output ========================
    if not is_power_on:
        set_led_pwm(RED_CH, 0); set_led_pwm(GREEN_CH, 0); set_led_pwm(BLUE_CH, 0)
    else:
        if current_sys_mode == 0: t, b = manual_temp, manual_brightness
        elif current_sys_mode == 1: t, b = 0.0, manual_brightness
        elif current_sys_mode == 2: t, b = 1.0, manual_brightness
        else:
            # Autonomous lighting adjustment based on physiological state
            target_lux = 600 if current_attention_state == "FOCUSED" else (250 if current_attention_state == "DISTRACTED" else 400)
            target_t = 1.0 if current_attention_state == "FOCUSED" else (0.0 if current_attention_state == "DISTRACTED" else 0.5)
            ai_temp += (target_t - ai_temp) * 0.02
            if current_lux > 0:
                err = target_lux - current_lux
                if abs(err) > 30: ai_brightness = max(0.1, min(1.0, ai_brightness + (0.005 if err > 0 else -0.005)))
            t, b = ai_temp, ai_brightness

        r = (COLOR_WARM[0] + (COLOR_COOL[0] - COLOR_WARM[0]) * t) * b
        g = (COLOR_WARM[1] + (COLOR_COOL[1] - COLOR_WARM[1]) * t) * b
        bl = (COLOR_WARM[2] + (COLOR_COOL[2] - COLOR_WARM[2]) * t) * b
        set_led_pwm(RED_CH, r); set_led_pwm(GREEN_CH, g); set_led_pwm(BLUE_CH, bl)

    # ======================== 8. UI Telemetry & Warnings ========================
    if not calibration_complete:
        cv2.putText(frame, f"CALIBRATING: {max(0, 5-elapsed):.1f}s", (int(w/2-150), int(h/2)), 1, 2, (0, 0, 255), 2)
    else:
        bpm = len(blink_times_window) 
        y = 30
        y = put_text_on_frame(frame, f"SYS: {SYS_MODES[current_sys_mode]} ({'BRIGHT' if adjust_param=='brightness' else 'TEMP'})", y, (255, 255, 255))
        y = put_text_on_frame(frame, f"EAR: {EAR:.3f}", y)
        y = put_text_on_frame(frame, f"BPM: {bpm}", y, (255, 255, 0))
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
            cv2.putText(frame, "DISTRACTED!", (w-200, 30), 1, 1.5, (0, 0, 255), 2)

    cv2.putText(frame, f"FPS: {int(fps)}", (w-120, h-20), 1, 1.2, (0, 255, 0), 2)
    cv2.imshow('EdgeHCL Adaptive System (Standard)', frame)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
mqtt_client.loop_stop()
set_led_pwm(RED_CH, 0); set_led_pwm(GREEN_CH, 0); set_led_pwm(BLUE_CH, 0)