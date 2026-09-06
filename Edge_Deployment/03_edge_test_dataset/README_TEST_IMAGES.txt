Raspberry Pi edge test images
================================
Unique final test images: 90
Smoke-test images: 10
Stored inference resolution: 640 x 640 pixels
Image format: JPEG
Naming format: EDGE_TEST_001_<original_name>_640x640.jpg through EDGE_TEST_090_<original_name>_640x640.jpg
Smoke-test naming: SMOKE_01_EDGE_TEST_... through SMOKE_10_EDGE_TEST_...
JPEG quality: 95
Mean file size: 257.13 KB
Minimum file size: 110.50 KB
Maximum file size: 321.54 KB
Total 640x640 test-image size: 22.60 MB

The original untouched test images are preserved separately. The 640x640 copies use
letterboxing with fill value 114, and their YOLO labels are transformed to match the
new canvas exactly. The complete per-image resolution and byte-size details are in:
07_manifests_checksums/edge_test_90_image_resolution_and_size_manifest.csv
