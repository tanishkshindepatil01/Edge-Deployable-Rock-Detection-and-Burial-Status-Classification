# Edge Deployable Rock Detection and Burial Status Classification

This project develops a lightweight computer-vision system for detecting rocks in agricultural field images and classifying them as either **completely exposed** or **half-buried**.

Three YOLO models were compared: **YOLOv8n, YOLO11n and YOLO11s**. YOLOv8n was selected using validation **mAP50–95** and was then deployed and benchmarked on a **Raspberry Pi 5** using multiple inference formats.

## Project Summary

- **Dataset:** 448 RGB images with 706 annotated rocks
- **Classes:** Completely exposed and half-buried rocks
- **Models compared:** YOLOv8n, YOLO11n and YOLO11s
- **Selected model:** YOLOv8n
- **Edge device:** Raspberry Pi 5
- **Deployment formats:** PyTorch FP32, ONNX FP32, TFLite FP32, TFLite INT8, NCNN FP32 and NCNN FP16
- **Main evaluation:** Accuracy, latency, FPS, CPU usage, memory usage and operational stability

## Repository Structure

Edge_Deployment/   - Raspberry Pi deployment files, models, scripts and results  
Root_Folder/       - Group report, individual reflection, Jupyter Notebook and presentation  
Visualization/     - CSV files, Tableau sheets and project visualisations

## Team 09

- Tanishk Nanasaheb Shinde
- Megh Kashilkar
- Sanskar Sanju Gade
- Jitendra Suwalka

