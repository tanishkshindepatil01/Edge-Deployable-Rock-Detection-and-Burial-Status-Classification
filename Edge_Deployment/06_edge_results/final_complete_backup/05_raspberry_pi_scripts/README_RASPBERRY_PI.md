# Rock Burial Raspberry Pi Deployment Package

## What is included

Six variants of the same selected YOLO detector:

1. PyTorch FP32 reference
2. ONNX FP32
3. LiteRT/TFLite FP32
4. LiteRT/TFLite INT8 calibrated with 200 original training images
5. NCNN FP32
6. NCNN FP16

The final edge dataset contains 90 unique test images stored at
640 x 640 pixels. A separate 10-image smoke test is included.

## Connect the Raspberry Pi to the MacBook

1. Flash 64-bit Raspberry Pi OS and enable SSH in Raspberry Pi Imager.
2. Connect the MacBook and Raspberry Pi to the same Wi-Fi or Ethernet network.
3. From Mac Terminal, test the connection:

   ssh <username>@rockpi.local

4. Transfer the extracted Pi-only package from the Mac:

   bash 05_raspberry_pi_scripts/copy_project_from_mac_to_pi.sh rockpi.local <username> <local_package_folder>

The MacBook is only the remote terminal and file-transfer machine. Inference must run on the Raspberry Pi CPU.

## Raspberry Pi setup

From the Raspberry Pi terminal:

   cd ~/rock_edge_project
   bash 05_raspberry_pi_scripts/setup_raspberry_pi.sh ~/rock_edge_project ~/rock_edge_venv
   source ~/rock_edge_venv/bin/activate
   python 05_raspberry_pi_scripts/verify_environment.py --project-root ~/rock_edge_project

## Ten-image functional smoke test

Use one format first, for example NCNN FP32:

   python 05_raspberry_pi_scripts/run_single_image.py \
     --model-folder 02_selected_model/05_ncnn_fp32 \
     --image 03_edge_test_dataset/03_smoke_test_10/images/<image_name>.jpg \
     --output-dir 06_edge_results/05_ncnn_fp32/smoke_test

Repeat for representative images and confirm boxes, class labels, counts, images and JSON outputs.

## Final benchmark across all six formats

   python 05_raspberry_pi_scripts/benchmark_deployment.py \
     --project-root ~/rock_edge_project \
     --warmup 10 \
     --passes 3 \
     --imgsz 640 \
     --conf 0.25 \
     --iou 0.7

This performs 10 unrecorded warm-up inferences for each format and
3 complete timed passes over all 90 test images.
Accuracy is computed on the 90 unique images; repeated passes are used only for stable timing and hardware measurements.

## Thirty-minute stability test

After choosing the best accuracy-efficiency format, run:

   python 05_raspberry_pi_scripts/stability_test.py \
     --project-root ~/rock_edge_project \
     --model-folder 05_ncnn_fp32 \
     --minutes 30

Replace `05_ncnn_fp32` with the actual winning folder.

## Visualization outputs

Notebook visualizations are stored in:

   visualization/

Raspberry Pi benchmark charts and specifically numbered annotated images are added to:

   visualization/08_raspberry_pi_runtime/

The benchmark script creates RP01–RP07 comparison plots. The stability script creates RP08–RP10 stability plots.

## Results to report

- Precision, recall, F1, mAP50 and mAP50-95
- Per-image rock-count error
- Model size
- Model-loading time
- Median, mean and P95 latency
- Median FPS
- CPU use
- Peak RAM
- Start, maximum and end temperature
- Throttling status
- Stability-test behaviour

Do not select the winner using speed alone. Select the format that preserves detection and half-buried-class accuracy while giving acceptable latency, memory use, temperature and reliability.
