# Y-MAX

**YOLO-based Multimodal Analysis eXplorer (Y-MAX)** is a Python computer-vision application that detects people in images and tracks people across video frames using Ultralytics [YOLO](https://docs.ultralytics.com/) and [ByteTrack](https://github.com/ifzhang/ByteTrack).

Current version: **2.0.0 — DEPTH UNLOCKED**

## Index

- [Features](#features)
- [Gallery](#gallery) (Coming soon)
- [Getting started](#getting-started)
- [Using Y-MAX](#using-y-max)
- [How it works](#how-it-works)
- [Important information](#important-information)
- [Changelog](CHANGELOG.md) (Separate file)
- [License](LICENSE.md) (Separate file)

## Features

- Person detection in images using YOLO.
- Person detection and tracking in videos using YOLO and ByteTrack.
- RGB processing using colour images and videos.
- RGB-D processing using paired colour and depth images, as well as supported RGB-D recordings.
- Distance estimation for detected people using depth information.
- Median distance measurements displayed alongside detection bounding boxes.
- Annotated colour images and videos, with tracking IDs where applicable.
- Annotated depth maps displaying YOLO detection bounding boxes and median distance measurements.
- Tracking IDs maintained across consecutive video frames.
- Pretrained YOLO26 model selection and custom model support.
- Adjustable confidence thresholds.
- Individual and batch media processing.
- Processing statistics, including detection counts, tracking counts, processing time, file sizes and video FPS.

## Gallery

*Coming soon!*

## Getting started

### Requirements

- [Python](https://www.python.org/) 3.13 or newer.
- [uv](https://docs.astral.sh/uv/).

### Running the program

1. Clone or download the [Y-MAX repository](https://github.com/max-legend-shady/Y-MAX).
2. Open a terminal in the `Y-MAX` project folder.
3. Install the project dependencies:

   ```
   uv sync
   ```

4. Start the program:

   ```
   uv run python y_max.py
   ```

### Project structure

Y-MAX includes the following files and folders. Folders not present in the repository are automatically created when `y_max.py` runs.

```text
Y-MAX/
├── input/
│   ├── images/
│   │   ├── RGB/
│   │   └── RGBD/
│   └── videos/
│       ├── RGB/
│       └── RGBD/
├── output/
│   ├── images/
│   │   ├── RGB/
│   │   └── RGBD/
│   └── videos/
│       ├── RGB/
│       └── RGBD/
├── models/
├── pyproject.toml
├── uv.lock
├── y_max.py
├── README.md
├── LICENSE.md
└── CHANGELOG.md
```

| File or folder | Use |
| --- | --- |
| `input/images/RGB/` | Input colour images. |
| `input/images/RGBD/` | Input RGB-D image pairs. |
| `input/videos/RGB/` | Input colour videos. |
| `input/videos/RGBD/` | Input RGB-D recordings. |
| `output/images/RGB/` | Annotated colour images. |
| `output/images/RGBD/` | Annotated RGB-D colour images and depth maps. |
| `output/videos/RGB/` | Annotated colour videos. |
| `output/videos/RGBD/` | Annotated RGB-D colour videos and depth-map videos. |
| `models/` | Pretrained model weights download location. |
| `pyproject.toml` and `uv.lock` | Necessary for `uv`'s functionality and dependency management. |
| `y_max.py` | Y-MAX's main script. |
| `README.md` | An extended description of Y-MAX. |
| `LICENSE.md` | A legal notice for the use of Y-MAX. |
| `CHANGELOG.md` | A list of all notable updates to Y-MAX. |

## Using Y-MAX

### Analysis workflow

1. **Select media type:** choose image processing or video processing.
2. **Choose processing mode:** select RGB (colour only) or RGB-D (colour + depth).
3. **Choose a model:** select a pretrained YOLO26 model or provide a custom `.pt` model.
4. **Set confidence:** choose a threshold between `0` and `1`.
5. **Select a folder:** choose an input folder or press Enter to use the default.
6. **Choose files:** enter file numbers separated by commas, or enter `all` to select every supported file found.
7. **Analyse:** wait for processing to finish and review the results and statistics.

For RGB-D images, Y-MAX automatically matches colour and depth files using their filenames. For example, `tum_person_01_color.png` must have a matching `tum_person_01_depth.png` file.

### Navigation and controls

| Input | Action |
| --- | --- |
| Menu option number | Selects the corresponding option. |
| Enter | Uses the displayed default at supported prompts. |
| `back` | Returns to the previous selection step. |
| `all` | Selects all supported files found. |
| `Ctrl+C` | Interrupts the current batch. Files completed before an interruption are saved, but incomplete video outputs are removed. |

For more detailed instructions, select **Read the instructions** in the main menu.

## How it works

### 1. Model selection

Y-MAX initialises its directories and configuration before displaying the main menu.

The user first selects image processing or video processing, followed by RGB or RGB-D mode. They then choose a pretrained YOLO26 model or a custom `.pt` model. The selected model is loaded through Ultralytics.

### 2. Media selection and configuration

The user sets a confidence threshold, which determines the minimum confidence required for a detection to be included.

Y-MAX then searches the selected input folder and its subfolders for supported media files. Users can select individual files or process all files found.

In RGB-D image mode, Y-MAX searches for matching colour and depth files. Both files must share the same filename prefix and use the `_color` and `_depth` suffixes respectively.

For example:

```text
tum_person_01_color.png
tum_person_01_depth.png
```

The two files are treated as one RGB-D image pair. During file selection, both filenames are displayed together.

### 3. Detection and tracking

Y-MAX uses YOLO to detect people, filtering detections using class index `0`.

- **Images:** Each image is analysed independently. Detected people are identified using bounding boxes and confidence scores.
- **Videos:** OpenCV reads the video frame by frame. YOLO detects people in each frame, while ByteTrack attempts to associate detections across consecutive frames. Successfully tracked detections receive tracking IDs.
- **RGB-D processing:** Colour images provide the visual information used for detection, while their corresponding depth data provides distance measurements. Y-MAX estimates the distance to each detected person using the median of valid depth measurements within a central region of their bounding box.

Tracking IDs are temporary labels, not real-world identities. They may change or disappear when people are occluded or difficult to detect. Tracking starts afresh for each video.

### 4. Output and statistics

Y-MAX saves annotated images and videos in the corresponding RGB or RGB-D output folders.

RGB outputs contain YOLO detection bounding boxes and, for videos, tracking IDs.

RGB-D outputs include both annotated colour media and annotated depth maps. Depth maps display the detection bounding boxes alongside the estimated median distance for each detected person.

Output filenames include the source name, selected model and confidence threshold. Existing outputs are not overwritten; a number is added to avoid filename conflicts.

Y-MAX also reports processing statistics:

| Statistic | Meaning |
| --- | --- |
| Total detections | All detections recorded. In videos, the same person can be counted across multiple frames. |
| Tracked detections | Detections that received a tracking ID. |
| Distinct track IDs | Different tracking IDs observed throughout a video. |
| Processed frames | Number of video frames analysed. |
| Processing time | Time spent analysing the media. |
| Average processing FPS | Number of frames processed per second. |
| Input/output size | File sizes before and after processing. |

## Important information

### Supported formats

| Media | Supported extensions or naming |
| --- | --- |
| Colour images | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp` |
| RGB-D colour images | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, using the `_color` suffix |
| RGB-D depth images | `.png`, using the matching `_depth` suffix |
| Colour videos | `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm` |
| RGB-D recordings | Supported RealSense `.db3` recordings |
| Custom models | `.pt` |

### RGB-D input requirements

RGB-D image pairs must meet the following requirements:

- Colour and depth images must have matching filename prefixes.
- Colour and depth images must have matching dimensions.
- Depth images must be aligned with their corresponding colour images.
- Depth images must use 16-bit PNG encoding.
- Depth values must use the measurement units expected by Y-MAX.

Y-MAX does not calibrate or align separately supplied RGB-D image pairs.

### Output naming

Output filenames identify the original media, selected model and confidence threshold, and whether it is an image (annotated) or video (tracked). RGB-D outputs additionally distinguish between colour and depth files using `_color` and `_depth` suffixes.

| Media type | Output file name |
| --- | --- |
| RGB Image | {original filename}_people-annotated_model-{model name}_confidence-{confidence threshold} |
| RGB Video | {original filename}_people-tracked_model-{model name}_confidence-{confidence threshold} |
| RGBD Image (Color) | {original filename}_people-annotated_model-{model name}_confidence-{confidence threshold}_color |
| RGBD Image (Depth map) | {original filename}_people-annotated_model-{model name}_confidence-{confidence threshold}_depth |
| RGBD Video (Color) | {original filename}_people-tracked_model-{model name}_confidence-{confidence threshold}_color |
| RGBD Video (Depth map) | {original filename}_people-tracked_model-{model name}_confidence-{confidence threshold}_depth |

If an output filename already exists, Y-MAX automatically adds a number (_{number starting at 2} at the end) to avoid overwriting it.

### Known limitations

- Y-MAX currently detects and tracks people only.
- Custom models are expected to use class index `0` for people.
- Detection and tracking accuracy depends on the selected model, confidence threshold and media quality.
- RGB-D image pairs must have matching filenames, using `_color` and `_depth` suffixes.
- Depth measurements depend on the quality and validity of the input depth data.
- Median distance measurements are estimates, not guaranteed exact distances.
- Tracking IDs can switch or disappear when people are occluded or difficult to detect.
- Tracking IDs are not maintained between separate videos.
- OpenCV-generated output videos do not preserve the source video's audio.
- Tracking statistics are not guaranteed counts of real-world individuals.

## Thank you for using Y-MAX!
