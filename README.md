# EdgeHCL: Adaptive Human-Centric Lighting via Heterogeneous Edge Inference

## Project Overview
EdgeHCL is an advanced Internet of Things (IoT) ecosystem designed to mitigate digital visual fatigue and enhance cognitive focus for long-duration display users. Unlike traditional wearable-based solutions (e.g., EEG or heart-rate monitors) that are intrusive and costly, EdgeHCL utilizes non-intrusive computer vision and heterogeneous edge computing to analyze physiological states in real-time. The system dynamically modulates environmental lighting parameters based on the user's Attention Visual Analyzer (AVA) score.

## Technical Architecture

### 1. Heterogeneous Inference Pipeline
The system implements a custom Finite State Machine (FSM) to decouple heavy computational tasks between the Neural Processing Unit (NPU) and Central Processing Unit (CPU).
* **Macro-Detection (NPU)**: Initial face localization is performed using an INT8-quantized BlazeFace model on the Hailo-8L NPU, ensuring high global-search efficiency.
* **Micro-Landmarking (CPU)**: Upon localization, the system crops a dynamic Region of Interest (ROI) and hands over the control to the CPU to execute a high-fidelity 478-point Face Mesh.
* **Coordinate Mapping**: Localized coordinates are re-mapped to global space via inverse transformation matrices to ensure spatial consistency.

### 2. Physiological Quantification Models
* **Eye Aspect Ratio (EAR)**: Quantifies ocular aperture to detect blink frequency.
* **Mouth Aspect Ratio (MAR)**: Monitors lip protrusion to detect yawning events.
* **3D Gaze Deviation**: Calculates the angular offset between the real-time nasal-mandibular vector and the calibrated reference vector.

### 3. Communication and Control
Data exchange between the inference engine (Raspberry Pi 5) and the lighting terminal is orchestrated via the MQTT protocol.

## Performance Benchmarks
The optimized heterogeneous architecture significantly reduces inference latency compared to standard CPU-only implementations.

| Metric | Pure CPU Baseline | Heterogeneous (NPU+CPU) |
| :--- | :--- | :--- |
| Inference Latency | 11.99 ms | 11.08 ms |
| Potential Throughput | 83.39 FPS | 90.21 FPS |

## Hardware Configuration
* **Edge Compute**: Raspberry Pi 5 + Hailo-8L AI Accelerator.
* **Control Node**: ESP32 Microcontroller.
* **Illumination Control**: PCA9685 PWM Controller + MOSFET Driver.
* **Sensing**: BH1750 Ambient Light Sensor.

## Installation and Deployment
Ensure the Hailo-8L driver and MediaPipe environment are properly configured before execution.

## Credits and References

### Core Inference Framework
This project utilizes and extends the inference pipeline developed by **AlbertaBeef**, which integrates MediaPipe with Hailo-8 acceleration. 
- **Reference Project**: [Accelerating MediaPipe Models with Hailo-8](https://www.hackster.io/AlbertaBeef/accelerating-the-mediapipe-models-with-hailo-8-24e037)
- **Contribution**: We have optimized the original sequential processing into a **Heterogeneous Finite State Machine (FSM)** and implemented the **Attention Visual Analyzer (AVA)** for real-time HCL control.

### Open Source Libraries
- **MediaPipe**: For high-fidelity face mesh and landmark extraction.
- **Hailo RT TAPPAS**: For NPU-accelerated face detection.
- **Adafruit libraries**: For PCA9685 and BH1750 hardware interfacing.
```bash
# Launch the main adaptive lighting pipeline
python main_hcl_system.py --model_path ./models/hailo8l_face.hef --mqtt_broker localhost
