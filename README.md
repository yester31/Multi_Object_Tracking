# Multi-Object Tracking with TensorRT

## 🚧 Project Status

This project is **under active development** and not yet fully complete.  

GOAL : 
This repository provides a **modular framework** for Multi-Object Tracking (MOT) using **various detection models and tracking algorithms** in **C++ with TensorRT**.  

<!-- 
It is designed for **real-time performance**, **scalability**, and easy integration in both research and production environments.
-->
---

## 📂 Project Structure

- **Reference**
  - Validation scripts directly based on tracking methods from the official GitHub repositories.  
  - Analysis of tracking algorithm implementations for easier understanding and further development.  

- **ONNX_Generator**  
  - Export PyTorch models to ONNX format  
  - Download pre-trained ONNX models  

- **Detector**  
  - Build TensorRT engine from ONNX  
  - Perform high-speed inference with TensorRT  
  - Includes preprocessing and postprocessing modules  

- **Tracker**  
  - Implement multiple tracking methods (e.g., ByteTrack, DeepSORT)  
  - Assign object IDs and estimate trajectories from detection results  

- **Run**  
  - Demo code integrating Detector and Tracker  
  - Input → Detection → Tracking → Visualization  

---

## 🚀 Workflow

1. **Model Preparation**  
   - Export PyTorch models to ONNX using `ONNX_Generator`  
   - Or download pre-trained ONNX models  

2. **TensorRT Engine Build**  
   - Convert ONNX models into TensorRT engines via `Detector`  
   - Cache and reuse engines for fast startup  

3. **Inference and Tracking**  
   - Run detection with `Detector`  
   - Apply tracking algorithms with `Tracker`  
   - Visualize results with integrated demo in `Run`  

---


<!--  

## ✅ Key Features

- **High Performance**: TensorRT optimization with FP16/INT8 support  
- **Modular Design**: Detection and tracking modules are independent and interchangeable  
- **Real-Time Ready**: Suitable for drones, autonomous driving, and surveillance systems  
- **Extensible**: Supports various detection models (YOLO, D-FINE, etc.) and tracking algorithms  

---

## 📌 Example Usage

```bash
# 1. Export or download ONNX model

# 2. Run demo with Detector + Tracker

-->