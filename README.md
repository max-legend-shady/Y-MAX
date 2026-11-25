# Y-MAX

**YOLO-based Multimodal Analysis eXplorer (Y-MAX)** is a Python computer-vision application that detects people in images and tracks people across video frames using Ultralytics [YOLO](https://docs.ultralytics.com/) and [ByteTrack](https://github.com/ifzhang/ByteTrack).

## Index

-   [Features](#features)
-   [Getting started](#getting-started)
-   [Using Y-MAX](#using-y-max)
-   [How it works](#how-it-works)
-   [Important information](#important-information)

## Features

-   Person detection in images using YOLO.

-   Person detection and tracking in videos using YOLO and ByteTrack.

-   Tracking IDs maintained across consecutive video frames.

-   Annotated image and video outputs.

-   Pretrained YOLO26 model selection and custom model support.

-   Adjustable confidence thresholds.

-   Individual and batch media processing.

-   Processing statistics, including detection counts, tracking counts, processing time, file sizes and video FPS.

## Getting started

### Requirements

-   [Python](https://www.python.org/) 3.13 or newer.

-   [uv](https://docs.astral.sh/uv/).    

### Running the program

1.  Clone or download the [Y-MAX repository](https://github.com/max-legend-shady/Y-MAX).

2.  Open a terminal in the `Y-MAX` project folder.

3.  Install the project dependencies:

    ```
    uv sync
    ```

4.  Start the program:

    ```
    uv run python y_max.py
    ```

### Project structure

Y-MAX includes the following files. Folders not present in the repository are automatically created when `y_max.py` runs.

   ```
   Y-MAX/
   ├── input/
   │   ├── images/
   │   └── videos/
   ├── output/
   │   ├── images/
   │   └── videos/
   ├── models/
   ├── pyproject.toml
   ├── uv.lock
   ├── y_max.py
   ├── README.md
   ├── LICENSE.md
   └── CHANGELOG.md
   ```

| File | Use |
| --- | --- |
| `input/images/` and `input/videos/`| Default input location |
| `output/images/` and `output/videos/`| Output location |
| `models/` | Pretrained model weights download location |
| `pyproject.toml` and `uv.lock` | Necessary for `uv`'s functionality |
| `y_max.py` | Y-MAX's main script |
| `README.md` | An extended description of Y-MAX |
| `LICENSE.md` | A legal notice for the use of Y-MAX |
| `CHANGELOG.md` | A list of all notable updates to Y-MAX |

## Using Y-MAX

### Analysis workflow

1.  **Select media type:** choose image detection or video tracking.

2.  **Choose a model:** select a pretrained YOLO26 model or provide a custom `.pt` model.

3.  **Set confidence:** choose a threshold between `0` and `1`.

4.  **Select a folder:** choose an input folder or press Enter to use the default.

5.  **Choose files:** enter file numbers separated by commas, or enter `all` to select every supported file found.

6.  **Analyse:** wait for processing to finish and review the results and statistics.

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

The user selects image detection or video tracking, then chooses a pretrained YOLO26 model or a custom `.pt` model. The selected model is loaded through Ultralytics.

### 2. Media selection and configuration

The user sets a confidence threshold, which determines the minimum confidence required for a detection to be included.

Y-MAX then searches the selected input folder and its subfolders for supported media files. Users can select individual files or process all files found.

### 3. Detection and tracking

Y-MAX uses YOLO to detect people, filtering detections using class index `0`.

- **Images:** Each image is analysed independently. Detected people are identified using bounding boxes and confidence scores.

- **Videos:** OpenCV reads the video frame by frame. YOLO detects people in each frame, while ByteTrack attempts to associate detections across consecutive frames. Successfully tracked detections receive tracking IDs.

Tracking IDs are temporary labels, not real-world identities. They may change or disappear when people are occluded or difficult to detect. Tracking starts afresh for each video.

### 4. Output and statistics

Y-MAX saves annotated images in `output/images/` and annotated videos in `output/videos/`. Video annotations include tracking IDs.

Output filenames include the source name, selected model and confidence threshold. Existing outputs are not overwritten; a number is added to avoid filename conflicts.

Y-MAX also reports processing statistics:

| Statistic | Meaning |
| --- | --- |
| Total detections | All detections recorded. In videos, the same  person can be counted across multiple frames.| 
| Tracked detections | Detections that received a tracking ID.| 
| Distinct track IDs | Different tracking IDs observed throughout a video.| 
| Processed frames | Number of video frames analysed.| 
| Processing time | Time spent analysing the media.| 
| Average processing FPS | Number of frames processed per second.| 
| Input/output size | File sizes before and after processing. | 

## Important information

### Supported formats

| Media | Supported extensions |
| --- | --- |
| Images | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp` |
| Videos | `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm` |
| Custom models | `.pt` |

### Known limitations

-   Y-MAX currently detects and tracks people only.

-   Custom models are expected to use class index `0` for people.

-   Detection and tracking accuracy depends on the selected model, confidence threshold and media quality.

-   Tracking IDs can switch or disappear when people are occluded or difficult to detect.

-   Tracking IDs are not maintained between separate videos.

-   OpenCV-generated output videos do not preserve the source video's audio.

-   Tracking statistics are not guaranteed counts of real-world individuals.

----------

**Thank you for using Y-MAX!**
