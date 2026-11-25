# Y-MAX Changelog

Notable changes to Y-MAX are documented here. Planned features are not part of the current release.

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

## Planned updates

-  **[2.0.0] — Multimodal analysis:** RGB-D depth information and related spatial analysis.

-  **[2.1.0] — Live feeds:** Live analysis of images/video received from cameras connected to your system.

*Note: These are development targets, not implemented features. Their scope may change.*
