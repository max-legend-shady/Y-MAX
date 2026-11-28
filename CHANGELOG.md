# Y-MAX Changelog

Notable changes to Y-MAX are documented here. This file records released functionality; planned features are not presented as implemented. Newest releases go at the top, previous release history stays intact.

## [2.0.0] — DEPTH UNLOCKED

### Added
- RGB-D image-pair analysis using aligned colour images and 16-bit depth PNG files.
- RealSense ROS 2 `.db3` recording playback using `pyrealsense2`, including depth-to-colour alignment.
- Median depth estimation for each detected person, sampled from the central region of the person's bounding box.
- Annotated colourised depth maps and depth-map videos, with YOLO bounding boxes and median-distance labels overlaid.
- ByteTrack-based person tracking in RGB and RGB-D videos.
- Batch statistics for RGB-D detections, valid and invalid distance estimates, tracked detections, track IDs, and processing FPS.

### Improved
- Reorganised input and output folders into `images` and `videos`, each containing `RGB` and `RGBD` folders.
- Reworked the selection flow so users choose images or videos first, then RGB or RGB-D.
- RGB-D image selection now displays each colour/depth pair together.
- RGB-D image pairs now use the matching filename convention `*_color.png` and `*_depth.png`.
- Updated the in-program instructions and README for the RGB-D workflows, folder structure, output files, and measurement limitations.
- Pretrained model weights are now correctly stored in the project's `models/` directory.
- Standardised output filenames across RGB and RGB-D image and video processing.
- Improved output conflict handling to prevent existing RGBD files from being overwritten.

### Known limitations
- Separately supplied RGB-D image pairs must already be aligned and have matching dimensions; Y-MAX does not calibrate or align them.
- Distance estimates are approximate camera-to-surface depth measurements, not full 3D positions or safety-rated measurements.
- Depth-map colours are visualisation only and use a fixed 0–5 metre range.
- RealSense `.db3` playback requires a compatible RealSense SDK build with ROS 2 rosbag2 playback support.
- Tracking IDs are temporary and may change or be lost; detections can include false positives and false negatives.
- OpenCV-generated output videos do not preserve source audio.
- Live camera streaming, robot navigation, and obstacle avoidance are not included.

## [1.2.0] — Codename: INITIATE TRACKING

### Added
- Video person tracking using Ultralytics YOLO and the ByteTrack tracking algorithm.
- Persistent tracking IDs across consecutive video frames.
- Tracking IDs displayed in annotated video outputs.
- Tracking statistics for individual videos: tracked detections and distinct track IDs.
- Batch tracking statistics that aggregate track instances across processed videos.

### Improved
- Video output filenames now include a tracking label to distinguish tracked videos from earlier detection-only outputs.
- Video reports now distinguish total detections across frames, tracked detections, and distinct track IDs.
- Tracking state is now reset when processing a new video, preventing tracking information from carrying over between separate videos.
- Instructions and README now explain tracking behaviour, statistics, navigation, and limitations.
- README now contains more information, including an index, a walkthrough of how the program works, a list of supported file types, and a list of limitations.

## [1.1.0] — Codename: MOTION DETECTED

### Added
- Video person detection, processed frame by frame, with annotated MP4 output.
- Per-file and batch statistics: elapsed time, detections, input/output sizes, completion counts, processed frames, and average processing FPS for videos.
- Batch interruption using `Ctrl+C`; previously completed outputs are retained.

### Improved
- Media selection now starts with a folder, searches its subfolders, and lets users select individual files or all supported files.
- Input validation, default-value handling, and cancellation paths throughout the selection workflow.
-  `back` navigation now returns to the immediately previous menu or selection step.
- Image and video output filenames include the source name, selected model, and confidence threshold, and existing outputs are not overwritten.
- Video processing validates source FPS and uses a configured fallback when the source FPS cannot be read.
- Instructions now describe the full workflow, controls, cancellation behaviour, outputs, statistics, and limitations.

## [1.0.0] — Codename: BOOT SEQUENCE

### Added
- The original [YOLO Person Detector](https://github.com/max-legend-shady/YOLO_Person_Detector) project's YOLO-based person-detection workflow, carried forward as an image-only application.
- Pretrained and custom model selection, confidence-threshold configuration, and annotated image outputs.
- Initial image detection results and person counts.

### Known limitation
- This version did not yet include robust input validation or the expanded navigation and cancellation handling introduced in v1.1.0.

## Planned updates

-  **[2.1.0] — Live feeds:** Live analysis of images/video received from cameras connected to your system.

*Note: These are development targets, not implemented features. Their scope may change.*
