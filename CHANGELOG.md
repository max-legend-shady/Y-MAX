# Y-MAX Changelog

Notable changes to Y-MAX are documented here. Planned features are not part of the current release.

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

-  **[1.2.0] — Object tracking:** persistent object IDs across video frames.

-  **[2.0.0] — Multimodal analysis:** RGB-D depth information and related spatial analysis.

*Note: These are development targets, not implemented features. Their scope may change.*