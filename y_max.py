# Import the libraries required for model loading, file handling, timing, and media processing
import math
import os
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
# Define the project directory and all input, output, and model folders
project_directory = Path(__file__).resolve().parent
input_images = project_directory / "input" / "images" / "RGB"
input_rgbd_images = project_directory / "input" / "images" / "RGBD"
input_videos = project_directory / "input" / "videos" / "RGB"
input_rgbd_videos = project_directory / "input" / "videos" / "RGBD"
output_images = project_directory / "output" / "images" / "RGB"
output_rgbd_images = project_directory / "output" / "images" / "RGBD"
output_videos = project_directory / "output" / "videos" / "RGB"
output_rgbd_videos = project_directory / "output" / "videos" / "RGBD"
models_folder = project_directory / "models"
# Create the folders required by Y-MAX if they do not already exist
for folder in (input_images, input_rgbd_images, input_videos, input_rgbd_videos, output_images, output_rgbd_images, output_videos, output_rgbd_videos, models_folder):
    folder.mkdir(parents=True, exist_ok=True)
# Define the supported file extensions for images, videos, recordings, and custom models
image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
rgbd_image_extensions = [".png"]
rgbd_recording_extensions = [".db3"]
model_extensions = [".pt"]
# Define the available pretrained YOLO models in the same order as the model menu
model_files = ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"]
# Define the project version and every default value used by an input or processing fallback
project_version = "2.0.0"
default_model = "yolo26n.pt"
default_threshold = 0.5
default_image_folder = input_images
default_video_folder = input_videos
default_rgbd_image_folder = input_rgbd_images
default_rgbd_video_folder = input_rgbd_videos
default_video_fps = 30.0
default_tracker = "bytetrack.yaml"
# Define the YOLO class index used to detect people
class_index = 0


def introduction():
    # Display the project name, version, and author
    print("\n----- WELCOME TO Y-MAX -----")
    print("YOLO-based Multimodal Analysis eXplorer")
    print(f"Version {project_version}")
    print("DEPTH UNLOCKED")
    print("\nCreated by Manuel Heredia")


def make_menu(title, options):
    # Display the menu title and add the project heading on the main menu
    if title == "MAIN MENU":
        print("---- Y-MAX ----")
    print(f"\n--- {title} ---")
    # Display each option with a number
    for number, option in enumerate(options, start=1):
        print(f"{number}. {option}")
    # Keep asking until the user selects a valid option
    while True:
        choice_text = input("Enter an option: ").strip()
        # Allow the user to type back when the current menu has a Back option
        if choice_text.lower().strip() == "back" and any(option.lower() == "back" for option in options):
            return next(number for number, option in enumerate(options, start=1) if option.lower() == "back")
        try:
            # Convert the numeric input into an integer after removing surrounding whitespace
            choice = int(choice_text.strip())
            # Return the choice only when it matches an available option
            if 1 <= choice <= len(options):
                return choice
            print(f"Invalid input. Please enter a number between 1 and {len(options)}.")
        except ValueError:
            # Explain how to correct non-numeric menu input
            print("Invalid input. Please enter a number or 'back' when available.")


def press_enter_to_continue():
    # Pause the program until the user presses Enter
    input("Enter to continue: ").lower().strip()


def clear_screen():
    # Clear the terminal using the command supported by the current operating system
    os.system("cls" if os.name == "nt" else "clear")


def format_bytes(byte_count):
    # Convert a byte count into a readable binary unit
    size = float(byte_count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024


def get_file_size(file_path):
    # Return the file size in bytes, or zero if the file cannot be accessed
    try:
        return file_path.stat().st_size
    except OSError:
        return 0


def print_batch_summary(media_label, batch_start, selected_count, completed_count, failed_count, interrupted_count, total_detections, total_frames, input_bytes, output_bytes, processing_seconds, total_tracked_detections=0, total_track_instances=0):
    # Calculate the total elapsed time, including model loading and processing
    total_elapsed = time.perf_counter() - batch_start
    print(f"\n--- {media_label.upper()} BATCH SUMMARY ---")
    # Report how many selected files completed, failed, or were interrupted
    print(f"\nFiles selected: {selected_count}")
    print(f"Files completed: {completed_count}")
    print(f"Files failed: {failed_count}")
    print(f"Files interrupted: {interrupted_count}")
    # Report the total detection count and data sizes
    print(f"\nTotal person detections: {total_detections}")
    print(f"Input data processed: {format_bytes(input_bytes)}")
    print(f"Output data saved: {format_bytes(output_bytes)}")
    # Report the total task duration and average processing rate
    print(f"\nTotal task time: {total_elapsed:.2f} seconds")
    print(f"Processing time: {processing_seconds:.2f} seconds")
    # Display video tracking statistics and processing performance
    if media_label == "video":
        print(f"Total tracked detections: {total_tracked_detections}")
        print(f"Track instances across videos: {total_track_instances}")
        if processing_seconds > 0:
            print(f"Average processing FPS: {total_frames / processing_seconds:.2f}")
            print(f"Total frames processed: {total_frames}")
    # Display the average processing rate for image batches
    elif media_label == "image" and processing_seconds > 0:
        print(f"Average image processing rate: {(completed_count + failed_count) / processing_seconds:.2f} images/second")


def get_pretrained_model_path(model_filename):
    # Build the full path inside the project's models folder
    local_model_path = models_folder / model_filename
    # Return the full path so Ultralytics downloads the model into models/
    return str(local_model_path.resolve())


def make_unique_output_path(folder, filename):
    # Create the output folder if it does not exist
    folder.mkdir(parents=True, exist_ok=True)
    # Prevent existing files from being overwritten
    output_path = folder / filename
    file_number = 2
    while output_path.exists():
        output_path = folder / f"{Path(filename).stem}_{file_number}{Path(filename).suffix}"
        file_number += 1
    return output_path


def make_unique_rgbd_output_paths(folder, color_filename, depth_filename):
    # Create the output folder if it does not exist
    folder.mkdir(parents=True, exist_ok=True)
    # Prepare the original paired output paths
    color_path = folder / color_filename
    depth_path = folder / depth_filename
    file_number = 2
    # Keep both output names matched when either file already exists
    while color_path.exists() or depth_path.exists():
        color_stem = Path(color_filename).stem.removesuffix("_color")
        depth_stem = Path(depth_filename).stem.removesuffix("_depth")
        color_path = folder / f"{color_stem}_{file_number}_color{Path(color_filename).suffix}"
        depth_path = folder / f"{depth_stem}_{file_number}_depth{Path(depth_filename).suffix}"
        file_number += 1
    # Return the two paths with a shared duplicate number
    return color_path, depth_path


def build_output_name(source_name, media_type, model_name, confidence_threshold, rgbd_suffix=None):
    # Choose the correct processing label for images or videos
    processing_label = "people-annotated" if media_type == "image" else "people-tracked"
    # Format the model name and confidence threshold consistently
    model_label = Path(model_name).stem.replace(" ", "-")
    confidence_label = f"{confidence_threshold:g}"
    # Add a colour or depth suffix only for RGB-D outputs
    modality_label = f"_{rgbd_suffix}" if rgbd_suffix else ""
    # Return the standardised output filename without its extension
    return f"{source_name}_{processing_label}_model-{model_label}_confidence-threshold-{confidence_label}{modality_label}"


def choose_model():
    # Keep the model menu available when the user backs out of custom-model entry
    while True:
        choice = make_menu("MODEL SELECTION", ["YOLO26 Nano (yolo26n.pt)", "YOLO26 Small (yolo26s.pt)", "YOLO26 Medium (yolo26m.pt)", "YOLO26 Large (yolo26l.pt)", "YOLO26 Extra Large (yolo26x.pt)", "Custom Model", "Back"])
        # Return the correctly indexed pretrained model selected by the user
        if 1 <= choice <= len(model_files):
            return get_pretrained_model_path(model_files[choice - 1])
        # Ask for a path when the user selects a custom model
        if choice == 6:
            while True:
                model_path_text = input(f"Enter the path to your custom model file (leave blank for {default_model}, or enter 'back'): ").strip().strip("&").strip().strip('"').strip("'")
                # Use the default pretrained model when the path is left blank
                if not model_path_text:
                    print(f"No input was entered, using default value {default_model}.")
                    return get_pretrained_model_path(default_model)
                # Return to the model menu when the user types back
                if model_path_text.lower().strip() == "back":
                    break
                # Convert the supplied path into a Path object
                model_path = Path(model_path_text).expanduser()
                # Resolve relative paths from the project directory
                if not model_path.is_absolute():
                    model_path = project_directory / model_path
                # Check that the custom model exists before checking its extension
                if not model_path.is_file():
                    print("No model file found at the specified path. Please enter a valid model file.")
                    continue
                # Reject model files that do not use a supported extension
                if model_path.suffix.lower() not in model_extensions:
                    print(f"Invalid input. Please select a supported model file ({', '.join(model_extensions)}).")
                    continue
                # Return the validated custom model path
                return str(model_path)
        # Return to the previous workflow when the user selects Back
        if choice == 7:
            return None


def choose_confidence_threshold():
    print("\n--- CONFIDENCE THRESHOLD SELECTION ---")
    # Keep asking until the user enters a valid confidence threshold or returns to model selection
    while True:
        threshold_text = input(f"Enter the confidence threshold (0-1, default {default_threshold}, or 'back'): ").strip()
        # Return to model selection when the user types back
        if threshold_text.lower().strip() == "back":
            return None
        # Use the configured default when the user presses Enter without a value
        if not threshold_text:
            print(f"No input was entered, using default value {default_threshold}.")
            return default_threshold
        try:
            # Convert the supplied value into a decimal number
            threshold = float(threshold_text)
            # Accept only values within the inclusive range from zero to one
            if 0 <= threshold <= 1 and math.isfinite(threshold):
                return threshold
            print("Invalid input. Please enter a number between 0 and 1.")
        except ValueError:
            # Explain how to correct non-numeric confidence input
            print("Invalid input. Please enter a number.")


def get_files(folder, extensions):
    # Recursively find supported files and sort them by their relative path
    available_files = sorted(file for file in folder.rglob("*") if file.is_file() and file.suffix.lower() in extensions)
    # Return the supported files to the selection menu
    return available_files


def choose_files(file_type):
    # Select the default folder, supported extensions, and labels for the requested media type
    if file_type == "image":
        default_folder = default_image_folder
        extensions = image_extensions
        file_label = "image"
        plural_label = "images"
        default_folder_label = "input/images/RGB"
    elif file_type == "video":
        default_folder = default_video_folder
        extensions = video_extensions
        file_label = "video"
        plural_label = "videos"
        default_folder_label = "input/videos/RGB"
    elif file_type == "rgbd_recording":
        default_folder = default_rgbd_video_folder
        extensions = rgbd_recording_extensions
        file_label = "RGB-D video recording"
        plural_label = "RGB-D video recordings"
        default_folder_label = "input/videos/RGBD"
    else:
        # Prevent unsupported internal file types from being processed
        raise ValueError("Unsupported file type. Use 'image', 'video', or 'rgbd_recording'.")
    print("\n--- FOLDER PATH SELECTION ---")
    # Return to the previous workflow menu when the user backs out of folder selection
    while True:
        folder_path_text = input(f"Enter the folder path (leave blank for {default_folder_label}, or enter 'back'): ").strip().strip("&").strip().strip('"').strip("'")
        # Return to the previous menu when the user types back at folder selection
        if folder_path_text.lower().strip() == "back":
            return None
        # Use the relevant default folder when the input is blank
        if not folder_path_text:
            folder_path = default_folder
            print(f"No input was entered, using default value {folder_path}.")
        else:
            # Convert the entered path into a Path object and resolve relative paths
            folder_path = Path(folder_path_text).expanduser()
            if not folder_path.is_absolute():
                folder_path = project_directory / folder_path
        # Check that the selected path points to an existing folder
        if not folder_path.is_dir():
            print("No folder found at the specified path. Please enter a valid folder path.")
            continue
        # Inform the user before searching through the folder and its subfolders
        print(f"\nSearching for supported {plural_label} in '{folder_path}' and its subfolders...")
        try:
            # Retrieve all supported files from the selected folder
            available_files = get_files(folder_path, extensions)
        except KeyboardInterrupt:
            # Allow the user to cancel a long recursive folder search
            print("Folder search cancelled. Returning to the previous menu.")
            return None
        except OSError as error:
            # Explain folder access errors and allow another folder to be selected
            print(f"Error searching the selected folder: {error}. Please choose another folder.")
            continue
        # Ask for another folder if no supported media files were found
        if not available_files:
            print(f"No supported {plural_label} found. Supported file types: {', '.join(extensions)}. Please select another folder.")
            continue
        # Display the files found in the selected folder
        print(f"\nFound {len(available_files)} supported {file_label}(s).")
        print(f"\n--- AVAILABLE {plural_label.upper()} ---")
        for number, file in enumerate(available_files, start=1):
            print(f"{number}. {file.relative_to(folder_path)}")
        # Keep the file list open until the user selects files or returns to folder selection
        while True:
            selection = input(f"Enter the {file_label} numbers separated by commas, 'all' to select every {file_label}, or 'back': ").lower().strip()
            # Return to folder selection when the user types back in the file list
            if selection == "back":
                break
            # Return every supported file when the user selects all
            if selection == "all":
                return available_files
            try:
                # Convert the comma-separated selections into integers
                selected_numbers = [int(number.strip()) for number in selection.split(",")]
                # Check that every selected number is within the displayed range
                if not selected_numbers or any(number < 1 or number > len(available_files) for number in selected_numbers):
                    print(f"Invalid input. Please enter {file_label} numbers between 1 and {len(available_files)}.")
                    continue
                # Remove duplicate selections while preserving the user's original order
                selected_numbers = list(dict.fromkeys(selected_numbers))
                # Return the selected file paths
                return [available_files[number - 1] for number in selected_numbers]
            except ValueError:
                # Explain the accepted file-selection formats
                print(f"Invalid input. Please enter {file_label} numbers separated by commas, 'all', or 'back'.")


def choose_rgbd_image_pairs():
    # Ask the user where the paired RGB-D images are stored
    print("\n--- RGB-D FOLDER PATH SELECTION ---")
    while True:
        folder_path_text = input(f"Enter the folder path (leave blank for {default_rgbd_image_folder}, or enter 'back'): ").strip().strip("&").strip().strip('"').strip("'")
        # Return to the previous menu when the user types back
        if folder_path_text.lower().strip() == "back":
            return None
        # Use the default RGB-D image folder when the input is blank
        if not folder_path_text:
            folder_path = default_rgbd_image_folder
            print(f"No input was entered, using default value {folder_path}.")
        else:
            folder_path = Path(folder_path_text).expanduser()
            if not folder_path.is_absolute():
                folder_path = project_directory / folder_path
        # Check that the selected path points to an existing folder
        if not folder_path.is_dir():
            print("No folder found at the specified path. Please enter a valid folder path.")
            continue
        # Find colour images using the required filename convention
        print(f"\nSearching for RGB-D image pairs in '{folder_path}' and its subfolders...")
        try:
            color_files = sorted(file for file in folder_path.rglob("*_color.png") if file.is_file())
        except OSError as error:
            print(f"Error searching the selected folder: {error}. Please choose another folder.")
            continue
        # Match each colour image with its corresponding depth image
        image_pairs = []
        for color_path in color_files:
            depth_path = color_path.with_name(f"{color_path.stem.removesuffix('_color')}_depth.png")
            if depth_path.is_file():
                image_pairs.append((color_path, depth_path))
        # Report an error if no complete pairs were found
        if not image_pairs:
            print("No RGB-D image pairs found. Please check the folder and filename conventions.")
            print("Each pair must contain a '_color.png' file and a matching '_depth.png' file.")
            continue
        # Display the available RGB-D image pairs
        print(f"\nFound {len(image_pairs)} complete RGB-D image pair(s).")
        for number, (color_path, depth_path) in enumerate(image_pairs, start=1):
            print(f"{number}. {color_path.relative_to(folder_path)} + {depth_path.relative_to(folder_path)}")
        # Keep the file list open until the user selects pairs or returns to folder selection
        while True:
            selection = input("Enter the RGB-D image numbers separated by commas, 'all' to select every pair, or 'back': ").lower().strip()
            if selection == "back":
                break
            if selection == "all":
                return image_pairs
            try:
                selected_numbers = [int(number.strip()) for number in selection.split(",")]
                if not selected_numbers or any(number < 1 or number > len(image_pairs) for number in selected_numbers):
                    print(f"Invalid input. Please enter RGB-D image numbers between 1 and {len(image_pairs)}.")
                    continue
                selected_numbers = list(dict.fromkeys(selected_numbers))
                return [image_pairs[number - 1] for number in selected_numbers]
            except ValueError:
                print("Invalid input. Please enter RGB-D image numbers separated by commas, 'all', or 'back'.")


def run_image_detection(model, model_name, image_path, confidence_threshold):
    # Run YOLO detection on the image and filter results to the person class
    results = model.predict(source=str(image_path), classes=[class_index], conf=confidence_threshold, verbose=False)
    # Retrieve the result for the single selected image
    result = results[0]
    # Count the detected people in the image
    detected_people = len(result.boxes)
    # Create an annotated image containing the detection boxes and labels
    annotated_image = result.plot()
    # Build a standardised output filename without overwriting existing results
    output_name = build_output_name(image_path.stem, "image", model_name, confidence_threshold)
    output_path = make_unique_output_path(output_images, f"{output_name}.jpg")
    # Save the annotated image and report an error if saving fails
    if not cv2.imwrite(str(output_path), annotated_image):
        raise OSError("Could not save the annotated image.")
    # Return the detection count and saved output path
    return detected_people, output_path


def run_image_analysis():
    # Repeat the setup stages so Back always returns to the immediately previous menu
    while True:
        model_name = choose_model()
        # Return to media selection when the user backs out of model selection
        if model_name is None:
            return False
        while True:
            confidence_threshold = choose_confidence_threshold()
            # Return to model selection when the user backs out of confidence selection
            if confidence_threshold is None:
                break
            selected_images = choose_files("image")
            # Return to confidence selection when the user backs out of folder selection
            if selected_images is None:
                continue
            # Start the full task timer before loading the selected model
            batch_start = time.perf_counter()
            print(f"\nLoading the selected YOLO model: {model_name}...")
            model_start = time.perf_counter()
            try:
                # Load the selected YOLO model
                model = YOLO(model_name)
            except KeyboardInterrupt:
                # Allow the user to cancel model loading
                print("Model loading cancelled. Returning to the previous menu.")
                return False
            except Exception as error:
                # Report model-loading errors without crashing the application
                print(f"Error loading the YOLO model: {error}. Please check the selected model and try again.")
                return True
            # Report how long model loading took
            print(f"Model loading time: {time.perf_counter() - model_start:.2f} seconds")
            # Initialise batch counters for successful, failed, and interrupted image processing
            completed_count = 0
            failed_count = 0
            interrupted_count = 0
            total_detections = 0
            input_bytes = 0
            output_bytes = 0
            processing_seconds = 0.0
            # Process each selected image and continue if an individual image fails
            for number, image_path in enumerate(selected_images, start=1):
                file_start = time.perf_counter()
                input_size = get_file_size(image_path)
                input_bytes += input_size
                print(f"\nProcessing image {number} of {len(selected_images)}: {image_path.name}...")
                try:
                    # Run detection and retrieve the count and output path
                    detected_people, output_path = run_image_detection(model, model_name, image_path, confidence_threshold)
                    # Calculate per-image processing time and output file size
                    file_elapsed = time.perf_counter() - file_start
                    output_size = get_file_size(output_path)
                    processing_seconds += file_elapsed
                    output_bytes += output_size
                    total_detections += detected_people
                    completed_count += 1
                    # Display detailed statistics for the completed image
                    print(f"Detected people: {detected_people}")
                    print(f"Processing time: {file_elapsed:.2f} seconds")
                    print(f"Input size: {format_bytes(input_size)}")
                    print(f"Output size: {format_bytes(output_size)}")
                    print(f"Annotated image saved to: {output_path}")
                except KeyboardInterrupt:
                    # Stop the remaining image queue while preserving every completed output
                    file_elapsed = time.perf_counter() - file_start
                    processing_seconds += file_elapsed
                    interrupted_count += 1
                    print("Image batch interrupted. Previously completed images remain saved; the current image was not saved.")
                    print(f"Time elapsed before interruption: {file_elapsed:.2f} seconds")
                    print(f"Input size: {format_bytes(input_size)}")
                    break
                except Exception as error:
                    # Record a failed image and continue with the remaining selected files
                    file_elapsed = time.perf_counter() - file_start
                    processing_seconds += file_elapsed
                    failed_count += 1
                    print(f"Error processing '{image_path.name}': {error}. Please check the image and try again.")
                    print(f"Time elapsed before failure: {file_elapsed:.2f} seconds")
                    print(f"Input size: {format_bytes(input_size)}")
            # Display the final batch report and output location
            print_batch_summary("image", batch_start, len(selected_images), completed_count, failed_count, interrupted_count, total_detections, 0, input_bytes, output_bytes, processing_seconds)
            print(f"Annotated images are saved in: {output_images}")
            return True


def run_video_detection(model, model_name, video_path, confidence_threshold):
    # Start timing the current video before opening it
    video_start = time.perf_counter()
    # Open the selected video and check that OpenCV can read it
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise OSError("Could not open the selected video.")
    # Read the video's frame rate and dimensions
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Stop if the video dimensions are unavailable or invalid
    if frame_width <= 0 or frame_height <= 0:
        capture.release()
        raise OSError("Could not read the video's frame dimensions.")
    # Use the configured fallback frame rate if the video's frame rate is invalid
    output_fps = source_fps
    if not math.isfinite(output_fps) or output_fps <= 0:
        output_fps = default_video_fps
        print(f"Video FPS could not be read. Using default value {default_video_fps} FPS.")
    # Build a standardised output filename without overwriting existing results
    output_name = build_output_name(video_path.stem, "video", model_name, confidence_threshold)
    output_path = make_unique_output_path(output_videos, f"{output_name}.mp4")
    # Create the video writer using the MP4V codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, (frame_width, frame_height))
    # Check that OpenCV successfully created the output video
    if not writer.isOpened():
        capture.release()
        writer.release()
        raise OSError("Could not create the output video.")
    # Initialise counters for processed frames, detections, and tracked detections
    frame_count = 0
    total_detections = 0
    tracked_detections = 0
    # Store every different tracking ID observed throughout the video
    unique_track_ids = set()
    try:
        # Read, analyse, annotate, and write each frame until the video ends or the user interrupts
        while True:
            success, frame = capture.read()
            # Stop when OpenCV reaches the end of the video
            if not success:
                break
            # Run person tracking on the current frame and preserve IDs between consecutive frames
            results = model.track(source=frame, classes=[class_index], conf=confidence_threshold, tracker=default_tracker, persist=frame_count > 0, verbose=False)
            result = results[0]
            # Add this frame's detections to the running total
            total_detections += len(result.boxes)
            # Retrieve the tracking IDs when the tracker has assigned them
            if result.boxes.id is not None:
                frame_track_ids = result.boxes.id.int().cpu().tolist()
                # Count every detection that received a tracking ID
                tracked_detections += len(frame_track_ids)
                # Add the current IDs to the set without duplicating existing IDs
                unique_track_ids.update(frame_track_ids)
            # Draw the detections and write the annotated frame
            writer.write(result.plot())
            # Update the processed-frame count and periodically report progress
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")
    except KeyboardInterrupt:
        # Release resources and remove the incomplete video when the user interrupts processing
        capture.release()
        writer.release()
        if output_path.exists():
            output_path.unlink()
        elapsed = time.perf_counter() - video_start
        print("Current video interrupted. Its incomplete output was removed; previously completed videos remain saved.")
        return False, frame_count, total_detections, tracked_detections, len(unique_track_ids), None, elapsed, output_fps
    except Exception:
        # Release resources and remove the incomplete output if processing fails
        capture.release()
        writer.release()
        if output_path.exists():
            output_path.unlink()
        raise
    # Release the input and output video resources after processing finishes
    capture.release()
    writer.release()
    # Reject videos from which no frames could be processed
    if frame_count == 0:
        if output_path.exists():
            output_path.unlink()
        raise OSError("No frames could be read from the selected video.")
    # Calculate processing time and output size after the video writer has closed
    elapsed = time.perf_counter() - video_start
    output_bytes = get_file_size(output_path)
    # Return the completion state, frame count, detections, output path, elapsed time, and output FPS
    return True, frame_count, total_detections, tracked_detections, len(unique_track_ids), output_path, elapsed, output_fps


def run_video_analysis():
    # Repeat the setup stages so Back always returns to the immediately previous menu
    while True:
        model_name = choose_model()
        # Return to media selection when the user backs out of model selection
        if model_name is None:
            return False
        while True:
            confidence_threshold = choose_confidence_threshold()
            # Return to model selection when the user backs out of confidence selection
            if confidence_threshold is None:
                break
            selected_videos = choose_files("video")
            # Return to confidence selection when the user backs out of folder selection
            if selected_videos is None:
                continue
            # Start the full task timer before loading the selected model
            batch_start = time.perf_counter()
            print(f"\nLoading the selected YOLO model: {model_name}...")
            model_start = time.perf_counter()
            try:
                # Load the selected YOLO model
                model = YOLO(model_name)
            except KeyboardInterrupt:
                # Allow the user to cancel model loading
                print("Model loading cancelled. Returning to the previous menu.")
                return False
            except Exception as error:
                # Report model-loading errors without crashing the application
                print(f"Error loading the YOLO model: {error}. Please check the selected model and try again.")
                return True
            # Report how long model loading took
            print(f"Model loading time: {time.perf_counter() - model_start:.2f} seconds")
            # Initialise batch counters for completed, failed, and interrupted video processing
            completed_count = 0
            failed_count = 0
            interrupted_count = 0
            total_detections = 0
            total_tracked_detections = 0
            total_track_instances = 0
            total_frames = 0
            input_bytes = 0
            output_bytes = 0
            processing_seconds = 0.0
            # Process each selected video and continue if an individual video fails
            for number, video_path in enumerate(selected_videos, start=1):
                video_attempt_start = time.perf_counter()
                input_size = get_file_size(video_path)
                input_bytes += input_size
                print(f"\nProcessing video {number} of {len(selected_videos)}: {video_path.name}...")
                try:
                    # Run detection and retrieve the completion state and video statistics
                    finished, frame_count, detections, tracked_detections, track_count, output_path, file_elapsed, output_fps = run_video_detection(model, model_name, video_path, confidence_threshold)
                    # Add the current video's processing time and statistics to the batch totals
                    processing_seconds += file_elapsed
                    total_frames += frame_count
                    total_detections += detections
                    total_tracked_detections += tracked_detections
                    total_track_instances += track_count
                    # Stop the remaining queue if the current video was interrupted
                    if not finished:
                        interrupted_count += 1
                        print(f"Frames processed before interruption: {frame_count}")
                        print(f"Detections before interruption: {detections}")
                        print(f"Tracked detections before interruption: {tracked_detections}")
                        print(f"Distinct track IDs before interruption: {track_count}")
                        print(f"Time elapsed before interruption: {file_elapsed:.2f} seconds")
                        print(f"Average processing FPS before interruption: {frame_count / file_elapsed:.2f}" if file_elapsed > 0 else "Average processing FPS: unavailable")
                        print(f"Input size: {format_bytes(input_size)}")
                        print("Output size: 0 B (incomplete video removed)")
                        break
                    # Record the completed video and its output size
                    output_size = get_file_size(output_path)
                    output_bytes += output_size
                    completed_count += 1
                    # Display detailed detection and tracking statistics for the completed video
                    print(f"Processed frames: {frame_count}")
                    print(f"Total detections across frames: {detections}")
                    print(f"Tracked detections: {tracked_detections}")
                    print(f"Distinct track IDs: {track_count}")
                    print(f"Processing time: {file_elapsed:.2f} seconds")
                    print(f"Average processing FPS: {frame_count / file_elapsed:.2f}" if file_elapsed > 0 else "Average processing FPS: unavailable")
                    print(f"Output video FPS: {output_fps:.2f}")
                    print(f"Input size: {format_bytes(input_size)}")
                    print(f"Output size: {format_bytes(output_size)}")
                    print(f"Annotated video saved to: {output_path}")
                except KeyboardInterrupt:
                    # Stop the remaining video queue while preserving every completed output
                    file_elapsed = time.perf_counter() - video_attempt_start
                    processing_seconds += file_elapsed
                    interrupted_count += 1
                    print("Video batch interrupted. Previously completed videos remain saved.")
                    print(f"Time elapsed before interruption: {file_elapsed:.2f} seconds")
                    print(f"Input size: {format_bytes(input_size)}")
                    break
                except Exception as error:
                    # Record the failed video's processing time and continue with the remaining files
                    file_elapsed = time.perf_counter() - video_attempt_start
                    processing_seconds += file_elapsed
                    failed_count += 1
                    print(f"Error processing '{video_path.name}': {error}. Please check the video and try again.")
                    print(f"Time elapsed before failure: {file_elapsed:.2f} seconds")
                    print(f"Input size: {format_bytes(input_size)}")
            # Display the final video batch report, including the tracking statistics
            print_batch_summary("video", batch_start, len(selected_videos), completed_count, failed_count, interrupted_count, total_detections, total_frames, input_bytes, output_bytes, processing_seconds, total_tracked_detections, total_track_instances)
            print(f"Annotated videos are saved in: {output_videos}")
            return True


def get_person_distances(result, depth_array, depth_scale):
    # Retrieve the bounding boxes produced by YOLO
    distances = []
    if result.boxes is None:
        return distances
    boxes = result.boxes.xyxy.cpu().numpy()
    # Estimate each person's distance using the central region of their bounding box
    for box in boxes:
        x1, y1, x2, y2 = [int(value) for value in box]
        # Keep the bounding box coordinates inside the depth image
        x1 = max(0, min(x1, depth_array.shape[1]))
        x2 = max(0, min(x2, depth_array.shape[1]))
        y1 = max(0, min(y1, depth_array.shape[0]))
        y2 = max(0, min(y2, depth_array.shape[0]))
        box_width = x2 - x1
        box_height = y2 - y1
        if box_width <= 0 or box_height <= 0:
            distances.append(None)
            continue
        # Use the central 50 percent of the bounding box to reduce background readings
        centre_x1 = x1 + box_width // 4
        centre_x2 = x2 - box_width // 4
        centre_y1 = y1 + box_height // 4
        centre_y2 = y2 - box_height // 4
        depth_region = depth_array[centre_y1:centre_y2, centre_x1:centre_x2]
        valid_depth = depth_region[depth_region > 0]
        if valid_depth.size == 0:
            distances.append(None)
            continue
        # Use the median depth to reduce the influence of outliers
        distances.append(float(np.median(valid_depth)) * depth_scale)
    return distances


def annotate_person_distances(image, result, distances):
    # Draw the YOLO detections and their labels onto a copy of the image
    annotated_image = result.plot()
    # Add the estimated distance beside each detected person
    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box, distance in zip(boxes, distances):
            x1, y1, x2, y2 = [int(value) for value in box]
            if distance is None:
                distance_label = "Distance unavailable"
            else:
                distance_label = f"{distance:.2f} m"
            # Place the distance label above the bounding box
            label_y = max(20, y1 - 10)
            cv2.putText(
                annotated_image,
                distance_label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
    return annotated_image


def create_depth_map(depth_array, depth_scale, maximum_depth=5.0):
    # Convert depth readings into a consistent 0–5 metre visual range
    depth_metres = depth_array.astype(np.float32) * depth_scale
    valid_depth = depth_array > 0
    depth_visual = np.zeros(depth_array.shape, dtype=np.uint8)
    depth_visual[valid_depth] = np.clip(
        depth_metres[valid_depth] * (255 / maximum_depth),
        0,
        255
    ).astype(np.uint8)
    # Apply an OpenCV colour map and keep invalid measurements black
    depth_map = cv2.applyColorMap(depth_visual, cv2.COLORMAP_JET)
    depth_map[~valid_depth] = (0, 0, 0)
    return depth_map


def annotate_depth_map(depth_map, result, distances):
    # Copy the colourised depth map before adding detection annotations
    annotated_depth_map = depth_map.copy()
    # Return the unchanged depth map when YOLO found no detections
    if result.boxes is None:
        return annotated_depth_map
    # Retrieve each detection's bounding box, class, and confidence
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    # Draw the bounding box, class, confidence, and distance for each person
    for box, class_id, confidence, distance in zip(boxes, classes, confidences, distances):
        x1, y1, x2, y2 = [int(value) for value in box]
        class_name = result.names[int(class_id)]
        # Format the distance or explain when no valid depth measurement exists
        distance_label = f"{distance:.2f} m" if distance is not None else "Distance unavailable"
        detection_label = f"{class_name} {confidence:.2f} | {distance_label}"
        # Draw a visible bounding box over the colourised depth map
        cv2.rectangle(annotated_depth_map, (x1, y1), (x2, y2), (255, 255, 255), 2)
        # Calculate the label dimensions and position
        label_y = max(22, y1 - 8)
        label_size, baseline = cv2.getTextSize(detection_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_top = max(0, label_y - label_size[1] - baseline - 4)
        # Add a dark background so the label remains readable over every depth-map colour
        cv2.rectangle(annotated_depth_map, (x1, label_top), (x1 + label_size[0] + 8, label_y + baseline), (0, 0, 0), -1)
        # Draw the person class, confidence score, and median-distance estimate
        cv2.putText(annotated_depth_map, detection_label, (x1 + 4, label_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    # Return the fully annotated depth map
    return annotated_depth_map


def process_rgbd_image_pair(color_path, depth_path, model, model_name, confidence_threshold):
    # Load the colour image
    color_image = cv2.imread(str(color_path))
    if color_image is None:
        raise OSError(f"Could not read the colour image '{color_path.name}'.")
    # Load the depth image without converting its original 16-bit values
    depth_array = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_array is None:
        raise OSError(f"Could not read the depth image '{depth_path.name}'.")
    # Validate the depth image format and dimensions
    if depth_array.dtype != np.uint16 or len(depth_array.shape) != 2:
        raise ValueError("Depth images must be single-channel 16-bit PNG files.")
    if depth_array.shape != color_image.shape[:2]:
        raise ValueError("The colour and depth images must have matching dimensions and be aligned.")
    # Run YOLO person detection on the colour image
    results = model.predict(
        source=color_image,
        classes=[class_index],
        conf=confidence_threshold,
        verbose=False
    )
    result = results[0]
    # Estimate each detected person's distance from the aligned depth image
    distances = get_person_distances(result, depth_array, 0.001)
    annotated_image = annotate_person_distances(color_image, result, distances)
    depth_map = create_depth_map(depth_array, 0.001)
    annotated_depth_map = annotate_depth_map(depth_map, result, distances)
    # Build standardised RGB-D output filenames
    image_name = color_path.stem.removesuffix("_color")
    color_output_name = build_output_name(image_name, "image", model_name, confidence_threshold, "color")
    depth_output_name = build_output_name(image_name, "image", model_name, confidence_threshold, "depth")
    annotated_path, depth_map_path = make_unique_rgbd_output_paths(
        output_rgbd_images,
        f"{color_output_name}.jpg",
        f"{depth_output_name}.png"
    )
    # Save both outputs
    if not cv2.imwrite(str(annotated_path), annotated_image):
        raise OSError("Could not save the annotated RGB-D image.")
    if not cv2.imwrite(str(depth_map_path), annotated_depth_map):
        if annotated_path.exists():
            annotated_path.unlink()
        raise OSError("Could not save the RGB-D depth map.")
    # Separate valid and invalid distance measurements
    valid_distances = [distance for distance in distances if distance is not None]
    return {
        "people_detected": len(distances),
        "valid_distances": len(valid_distances),
        "invalid_distances": len(distances) - len(valid_distances),
        "distances": valid_distances,
        "annotated_path": annotated_path,
        "depth_map_path": depth_map_path
    }


def run_rgbd_image_analysis():
    # Repeat setup stages so Back returns to the immediately previous menu
    while True:
        model_name = choose_model()
        if model_name is None:
            return False
        while True:
            confidence_threshold = choose_confidence_threshold()
            if confidence_threshold is None:
                break
            image_pairs = choose_rgbd_image_pairs()
            if image_pairs is None:
                continue
            # Load the selected YOLO model once for the complete batch
            print(f"\nLoading the selected YOLO model: {model_name}...")
            model_start = time.perf_counter()
            try:
                model = YOLO(model_name)
            except KeyboardInterrupt:
                print("Model loading cancelled. Returning to the previous menu.")
                return False
            except Exception as error:
                print(f"Error loading the YOLO model: {error}. Please check the selected model and try again.")
                return True
            print(f"Model loading time: {time.perf_counter() - model_start:.2f} seconds")
            # Initialise batch statistics
            batch_start = time.perf_counter()
            successful_images = 0
            failed_images = 0
            total_people = 0
            total_valid_distances = 0
            total_invalid_distances = 0
            all_distances = []
            # Process every selected RGB-D image pair
            for number, (color_path, depth_path) in enumerate(image_pairs, start=1):
                print(f"\nProcessing RGB-D image {number}/{len(image_pairs)}...")
                print(f"Colour image: {color_path.name}")
                print(f"Depth image: {depth_path.name}")
                try:
                    statistics = process_rgbd_image_pair(
                        color_path,
                        depth_path,
                        model,
                        model_name,
                        confidence_threshold
                    )
                    successful_images += 1
                    total_people += statistics["people_detected"]
                    total_valid_distances += statistics["valid_distances"]
                    total_invalid_distances += statistics["invalid_distances"]
                    all_distances.extend(statistics["distances"])
                    print(f"People detected: {statistics['people_detected']}")
                    print(f"Valid distance estimates: {statistics['valid_distances']}")
                    print(f"Invalid distance estimates: {statistics['invalid_distances']}")
                    print(f"Annotated image saved to: {statistics['annotated_path']}")
                    print(f"Depth map saved to: {statistics['depth_map_path']}")
                except KeyboardInterrupt:
                    print("RGB-D image batch interrupted. Previously completed outputs remain saved.")
                    break
                except Exception as error:
                    failed_images += 1
                    print(f"Error processing '{color_path.name}': {error}. Please check the image pair and try again.")
            # Display the final batch summary
            print("\n--- RGB-D IMAGE ANALYSIS SUMMARY ---")
            print(f"Image pairs selected: {len(image_pairs)}")
            print(f"Image pairs completed: {successful_images}")
            print(f"Image pairs failed: {failed_images}")
            print(f"Total people detected: {total_people}")
            print(f"Valid distance estimates: {total_valid_distances}")
            print(f"Invalid distance estimates: {total_invalid_distances}")
            if all_distances:
                print(f"Average distance: {np.mean(all_distances):.2f} m")
                print(f"Minimum distance: {np.min(all_distances):.2f} m")
                print(f"Maximum distance: {np.max(all_distances):.2f} m")
            else:
                print("No valid distance estimates were available.")
            print(f"\nAnnotated images are saved in: {output_rgbd_images}")
            print(f"Depth maps are saved in: {output_rgbd_images}")
            print(f"Total task time: {time.perf_counter() - batch_start:.2f} seconds")
            return True


def run_rgbd_recording_detection(recording_path, model, model_name, confidence_threshold):
    # Import the RealSense SDK only when recording playback is requested
    try:
        import pyrealsense2 as rs
    except ImportError as error:
        raise ImportError("pyrealsense2 is not installed. Run 'uv sync' and try again.") from error
    # Initialise the RealSense playback pipeline
    pipeline = rs.pipeline()
    configuration = rs.config()
    configuration.enable_device_from_file(str(recording_path), repeat_playback=False)
    writer = None
    depth_writer = None
    annotated_path = None
    depth_map_path = None
    interrupted = False
    frame_count = 0
    total_detections = 0
    total_tracked_detections = 0
    unique_track_ids = set()
    valid_distances = []
    invalid_distance_count = 0
    processing_start = time.perf_counter()
    output_fps = default_video_fps
    processing_error = None
    try:
        # Start playback and retrieve the recorded camera properties
        profile = pipeline.start(configuration)
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)
        align = rs.align(rs.stream.color)
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        if not math.isfinite(depth_scale) or depth_scale <= 0:
            raise OSError("The recording does not provide a valid depth scale.")
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        frame_width = color_profile.width()
        frame_height = color_profile.height()
        output_fps = float(color_profile.fps())
        if not math.isfinite(output_fps) or output_fps <= 0:
            output_fps = default_video_fps
            print(f"Recording FPS could not be read. Using default value {default_video_fps} FPS.")
        if frame_width <= 0 or frame_height <= 0:
            raise OSError("Could not read the recorded colour stream dimensions.")
        # Build standardised RGB-D colour and depth video filenames
        recording_name = recording_path.stem
        color_output_name = build_output_name(recording_name, "video", model_name, confidence_threshold, "color")
        depth_output_name = build_output_name(recording_name, "video", model_name, confidence_threshold, "depth")
        annotated_path, depth_map_path = make_unique_rgbd_output_paths(
            output_rgbd_videos,
            f"{color_output_name}.mp4",
            f"{depth_output_name}.mp4"
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(annotated_path), fourcc, output_fps, (frame_width, frame_height))
        depth_writer = cv2.VideoWriter(str(depth_map_path), fourcc, output_fps, (frame_width, frame_height))
        if not writer.isOpened() or not depth_writer.isOpened():
            raise OSError("Could not create one or both RGB-D output videos.")
        # Read the finite recording until playback reaches its end
        while True:
            frames = pipeline.poll_for_frames()
            if not frames:
                if playback.current_status() == rs.playback_status.stopped:
                    break
                time.sleep(0.001)
                continue
            # Align depth to the colour camera before measuring distances
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())
            depth_array = np.asanyarray(depth_frame.get_data())
            # Run YOLO and ByteTrack on the current colour frame
            results = model.track(
                source=color_image,
                classes=[class_index],
                conf=confidence_threshold,
                tracker=default_tracker,
                persist=frame_count > 0,
                verbose=False
            )
            result = results[0]
            # Estimate distances using the aligned depth frame and sensor depth scale
            distances = get_person_distances(result, depth_array, depth_scale)
            annotated_image = annotate_person_distances(color_image, result, distances)
            depth_map = create_depth_map(depth_array, depth_scale)
            annotated_depth_map = annotate_depth_map(depth_map, result, distances)
            writer.write(annotated_image)
            depth_writer.write(annotated_depth_map)
            # Update detection, tracking, and distance statistics
            total_detections += len(result.boxes) if result.boxes is not None else 0
            valid_frame_distances = [distance for distance in distances if distance is not None]
            valid_distances.extend(valid_frame_distances)
            invalid_distance_count += len(distances) - len(valid_frame_distances)
            if result.boxes is not None and result.boxes.id is not None:
                frame_track_ids = result.boxes.id.int().cpu().tolist()
                total_tracked_detections += len(frame_track_ids)
                unique_track_ids.update(frame_track_ids)
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} RGB-D frames...")
    except KeyboardInterrupt:
        interrupted = True
        print("Recording processing interrupted. The incomplete outputs will be removed.")
    except Exception as error:
        processing_error = error
    finally:
        # Release video writers and stop the RealSense pipeline
        if writer is not None:
            writer.release()
        if depth_writer is not None:
            depth_writer.release()
        try:
            pipeline.stop()
        except Exception:
            pass
    # Remove partial files after resources have been released
    if processing_error is not None:
        if annotated_path is not None and annotated_path.exists():
            annotated_path.unlink()
        if depth_map_path is not None and depth_map_path.exists():
            depth_map_path.unlink()
        raise processing_error
    # Remove incomplete output videos after interruption or an empty recording
    if interrupted or frame_count == 0:
        if annotated_path is not None and annotated_path.exists():
            annotated_path.unlink()
        if depth_map_path is not None and depth_map_path.exists():
            depth_map_path.unlink()
    elapsed = time.perf_counter() - processing_start
    return {
        "completed": not interrupted and frame_count > 0,
        "interrupted": interrupted,
        "frames": frame_count,
        "detections": total_detections,
        "tracked_detections": total_tracked_detections,
        "track_instances": len(unique_track_ids),
        "valid_distances": len(valid_distances),
        "invalid_distances": invalid_distance_count,
        "distances": valid_distances,
        "processing_seconds": elapsed,
        "output_fps": output_fps,
        "annotated_path": annotated_path if not interrupted and frame_count > 0 else None,
        "depth_map_path": depth_map_path if not interrupted and frame_count > 0 else None
    }


def run_rgbd_recording_analysis():
    # Repeat setup stages so Back returns to the immediately previous menu
    while True:
        model_name = choose_model()
        if model_name is None:
            return False
        while True:
            confidence_threshold = choose_confidence_threshold()
            if confidence_threshold is None:
                break
            selected_recordings = choose_files("rgbd_recording")
            if selected_recordings is None:
                continue
            # Load the selected YOLO model once for the complete batch
            print(f"\nLoading the selected YOLO model: {model_name}...")
            model_start = time.perf_counter()
            try:
                model = YOLO(model_name)
            except KeyboardInterrupt:
                print("Model loading cancelled. Returning to the previous menu.")
                return False
            except Exception as error:
                print(f"Error loading the YOLO model: {error}. Please check the selected model and try again.")
                return True
            print(f"Model loading time: {time.perf_counter() - model_start:.2f} seconds")
            # Initialise batch statistics
            batch_start = time.perf_counter()
            completed_count = 0
            failed_count = 0
            interrupted_count = 0
            total_frames = 0
            total_detections = 0
            total_tracked_detections = 0
            total_track_instances = 0
            total_valid_distances = 0
            total_invalid_distances = 0
            all_distances = []
            # Process each selected RealSense recording
            for number, recording_path in enumerate(selected_recordings, start=1):
                print(f"\nProcessing RGB-D recording {number}/{len(selected_recordings)}: {recording_path.name}")
                print("Loading the recorded colour and depth streams...")
                try:
                    statistics = run_rgbd_recording_detection(
                        recording_path,
                        model,
                        model_name,
                        confidence_threshold
                    )
                    total_frames += statistics["frames"]
                    total_detections += statistics["detections"]
                    total_tracked_detections += statistics["tracked_detections"]
                    total_track_instances += statistics["track_instances"]
                    total_valid_distances += statistics["valid_distances"]
                    total_invalid_distances += statistics["invalid_distances"]
                    all_distances.extend(statistics["distances"])
                    if statistics["interrupted"]:
                        interrupted_count += 1
                        print("RGB-D recording interrupted. Previously completed recordings remain saved.")
                        break
                    if not statistics["completed"]:
                        failed_count += 1
                        print("No usable RGB-D frames were found in this recording.")
                        continue
                    completed_count += 1
                    print(f"Frames processed: {statistics['frames']}")
                    print(f"Person detections: {statistics['detections']}")
                    print(f"Tracked detections: {statistics['tracked_detections']}")
                    print(f"Distinct track IDs: {statistics['track_instances']}")
                    print(f"Valid distance estimates: {statistics['valid_distances']}")
                    print(f"Invalid distance estimates: {statistics['invalid_distances']}")
                    print(f"Processing time: {statistics['processing_seconds']:.2f} seconds")
                    print(f"Average processing FPS: {statistics['frames'] / statistics['processing_seconds']:.2f}" if statistics["processing_seconds"] > 0 else "Average processing FPS: unavailable")
                    print(f"Annotated video saved to: {statistics['annotated_path']}")
                    print(f"Depth-map video saved to: {statistics['depth_map_path']}")
                except KeyboardInterrupt:
                    interrupted_count += 1
                    print("RGB-D recording batch interrupted. Previously completed recordings remain saved.")
                    break
                except Exception as error:
                    failed_count += 1
                    print(f"Error processing '{recording_path.name}': {error}. Check the recording and RealSense SDK version.")
            # Display the final batch summary
            print("\n--- RGB-D RECORDING ANALYSIS SUMMARY ---")
            print(f"Recordings selected: {len(selected_recordings)}")
            print(f"Recordings completed: {completed_count}")
            print(f"Recordings failed: {failed_count}")
            print(f"Recordings interrupted: {interrupted_count}")
            print(f"Total frames processed: {total_frames}")
            print(f"Total person detections: {total_detections}")
            print(f"Tracked detections: {total_tracked_detections}")
            print(f"Track instances: {total_track_instances}")
            print(f"Valid distance estimates: {total_valid_distances}")
            print(f"Invalid distance estimates: {total_invalid_distances}")
            if all_distances:
                print(f"Average distance: {np.mean(all_distances):.2f} m")
                print(f"Minimum distance: {np.min(all_distances):.2f} m")
                print(f"Maximum distance: {np.max(all_distances):.2f} m")
            else:
                print("No valid distance estimates were available.")
            print(f"\nAnnotated RGB-D videos are saved in: {output_rgbd_videos}")
            print(f"Depth-map videos are saved in: {output_rgbd_videos}")
            print(f"Total task time: {time.perf_counter() - batch_start:.2f} seconds")
            return True


def initiate_detection():
    # Let the user choose the media type before choosing RGB or RGB-D
    while True:
        media_choice = make_menu("MEDIA TYPE SELECTION", ["Image(s)", "Video(s)", "Back"])
        if media_choice == 3:
            print("Returning to the main menu...")
            return False
        media_type = "image" if media_choice == 1 else "video"
        modality_choice = make_menu("IMAGE TYPE SELECTION" if media_type == "image" else "VIDEO TYPE SELECTION", ["RGB (Colour only)", "RGB-D (Colour + Depth)", "Back"])
        if modality_choice == 3:
            continue
        # Start the selected RGB or RGB-D workflow
        if media_type == "image" and modality_choice == 1:
            print("Preparing RGB image analysis...")
            if run_image_analysis():
                return True
        elif media_type == "image" and modality_choice == 2:
            print("Preparing RGB-D image analysis...")
            if run_rgbd_image_analysis():
                return True
        elif media_type == "video" and modality_choice == 1:
            print("Preparing RGB video analysis...")
            if run_video_analysis():
                return True
        elif media_type == "video" and modality_choice == 2:
            print("Preparing RGB-D video analysis...")
            if run_rgbd_recording_analysis():
                return True


def read_instructions():
    # Display setup instructions and all RGB and RGB-D analysis workflows
    print("\n--- INSTRUCTIONS ---")
    print("\n--- STEP 1: PREPARATION ---")
    print("Install Python 3.13 or newer and uv, then run 'uv sync' from the Y-MAX project folder.")
    print("Y-MAX creates the required input, output, and models folders automatically.")
    print("RGB-D video playback requires the RealSense SDK and pyrealsense2 package included in the project dependencies.")
    print("The installed RealSense SDK must support native ROS 2 .db3 playback.")
    print("A RealSense camera is not required to analyse an existing compatible recording.")
    print("\n--- STEP 2: INPUT FOLDERS AND FILE FORMATS ---")
    print("Place RGB images in 'input/images/RGB' and RGB-D image pairs in 'input/images/RGBD'.")
    print("Place RGB videos in 'input/videos/RGB' and RealSense .db3 recordings in 'input/videos/RGBD'.")
    print("You can also enter a different folder when prompted.")
    print(f"RGB image formats: {', '.join(image_extensions)}.")
    print(f"RGB video formats: {', '.join(video_extensions)}.")
    print(f"RGB-D image format: {', '.join(rgbd_image_extensions)}.")
    print(f"RGB-D video recording format: {', '.join(rgbd_recording_extensions)}.")
    print("\n--- STEP 3: RGB-D IMAGE PAIR FORMAT ---")
    print("Every RGB-D image pair must contain one colour image and one matching depth image.")
    print("Use the filename convention 'scene_01_color.png' and 'scene_01_depth.png'.")
    print("The filename stem before '_color' and '_depth' must be identical.")
    print("Colour and depth images must have identical dimensions and already be aligned.")
    print("Depth images must be single-channel 16-bit PNG files with depth values in millimetres.")
    print("A depth value of zero represents an invalid or unavailable measurement.")
    print("Y-MAX does not align or calibrate separately supplied RGB-D image pairs.")
    print("\n--- STEP 4: MODEL AND CONFIDENCE ---")
    print("Choose YOLO26 Nano, Small, Medium, Large, Extra Large, or a custom '.pt' model.")
    print(f"When entering a custom model path, press Enter to use the default model ({default_model}).")
    print("Pretrained weights are loaded from the 'models' folder when present; otherwise Ultralytics may download them.")
    print(f"Enter a confidence threshold between 0 and 1. Press Enter to use the default value {default_threshold}.")
    print("Only person detections with confidence equal to or greater than the selected threshold are included.")
    print("Y-MAX assumes class index 0 represents people in the selected model.")
    print("\n--- STEP 5: RGB IMAGE AND VIDEO ANALYSIS ---")
    print("Choose Images or Videos first, then choose RGB (colour only) or RGB-D (colour + depth).")
    print("RGB images are processed individually using YOLO object detection.")
    print("RGB videos are processed frame by frame using YOLO and the ByteTrack tracking algorithm.")
    print(f"The configured tracking algorithm is '{default_tracker}'.")
    print("Tracking assigns temporary IDs to detected people and attempts to preserve them between consecutive frames.")
    print("Tracking IDs are not real-world identities and may change or be lost.")
    print("\n--- STEP 6: RGB-D IMAGE ANALYSIS ---")
    print("Y-MAX detects people in each colour image and samples the central region of each detection in the aligned depth image.")
    print("The median valid depth value in that region is used as the person's distance estimate.")
    print("Depth values are converted from millimetres to metres.")
    print("Y-MAX saves an annotated colour image and an annotated colourised depth map for every successfully processed pair.")
    print("Both outputs include the YOLO detection boxes and each person's median-distance label.")
    print("Depth-map colours use a fixed visual range of 0 to 5 metres; values beyond that range are clipped.")
    print("The depth-map colours are for visualisation only. Use the original depth values for measurements.")
    print("\n--- STEP 7: RGB-D VIDEO ANALYSIS ---")
    print("Y-MAX plays back RealSense .db3 recordings using the pyrealsense2 SDK.")
    print("Depth frames are aligned to the recorded colour stream before distance estimation.")
    print("The depth scale is read from the recording's RealSense device metadata.")
    print("YOLO and ByteTrack detect and track people across the recorded frames.")
    print("Y-MAX saves an annotated colour video and an annotated colourised depth-map video.")
    print("Both videos include the detection boxes and median-distance labels.")
    print("The recording is processed as fast as the computer allows; output video timing uses the recorded colour-stream FPS.")
    print("Native .db3 playback requires a RealSense SDK build with ROS 2 rosbag2 playback support.")
    print("\n--- STEP 8: RESULTS AND STATISTICS ---")
    print("RGB annotated images are saved in 'output/images/RGB'.")
    print("RGB-D annotated images and depth maps are saved in 'output/images/RGBD'.")
    print("RGB annotated videos are saved in 'output/videos/RGB'.")
    print("RGB-D annotated videos and depth-map videos are saved in 'output/videos/RGBD'.")
    print("Existing output files are not overwritten; a number is added to avoid filename conflicts.")
    print("Distance statistics include only detections with valid depth measurements.")
    print("A distance estimate is based on the median depth in the central part of a person's bounding box.")
    print("Measurements can be inaccurate around occlusions, reflective surfaces, edges, missing depth, or misalignment.")
    print("The distance estimate is camera-to-surface depth, not a complete 3D position or guaranteed person-to-camera range.")
    print("\n--- STEP 9: CANCELLATION AND LIMITATIONS ---")
    print("Type 'back' at supported selection prompts to return to the previous menu.")
    print("Press Ctrl+C during processing to stop the current batch.")
    print("Outputs completed before interruption remain saved; incomplete RGB-D recording videos are removed.")
    print("Detection and tracking may produce false positives, false negatives, or ID switches.")
    print("OpenCV-generated output videos do not preserve source audio.")
    print("Y-MAX does not provide live camera streaming, robot navigation, obstacle avoidance, or safety-rated distance measurements.")
    print("\nThank you for using Y-MAX!")


def show_credits():
    # Display the project name, current version, and author
    print("\n--- CREDITS ---")
    print("Y-MAX: YOLO-based Multimodal Analysis eXplorer")
    print("DEPTH UNLOCKED")
    print(f"Version {project_version}")
    print("Program written by Manuel Heredia")
    # Display versions of the direct dependencies used by this project
    print("\nDirect dependencies:")
    for package_name in ("ultralytics", "opencv-python", "numpy", "pyrealsense2", "lap"):
        try:
            # Retrieve the installed version of the current dependency
            package_version = version(package_name)
            print(f"- {package_name} ({package_version})")
        except PackageNotFoundError:
            # Report a dependency that is not installed in the active environment
            print(f"- {package_name} (not installed)")
    # Describe the educational purpose of the project
    print("\nThis program was made as a personal project to learn more about computer vision, object detection, and multi-object tracking.")
    print("I am also going to write an article on this project, so stay tuned!")
    print("If you have any questions or feedback, please feel free to reach out to me at the following email: manu.heredia.estrada@gmail.com")
    print("\nThank you for using Y-MAX!")


def confirm_exit():
    # Keep asking until the user confirms or declines the exit
    while True:
        confirmation = input("Enter 'yes' to exit, or leave blank to return to the main menu: ").lower().strip()
        # Confirm the exit when the user enters yes
        if confirmation in ("yes", "y"):
            return True
        # Return to the main menu when the user leaves the prompt blank
        if not confirmation:
            return False
        # Explain how to correct an unsupported confirmation
        print("Invalid input. Please enter 'yes' or leave the input blank.")



def main():
    # Clear the terminal and display the introduction
    clear_screen()
    introduction()
    # Keep displaying the main menu until the user confirms exit
    while True:
        choice = make_menu("MAIN MENU", ["Detect people in RGB or RGB-D images and videos", "Read the instructions", "Show the credits", "Exit"])
        # Start the image or video detection workflow and pause only after a result screen is ready
        if choice == 1:
            workflow_finished = initiate_detection()
            if workflow_finished:
                press_enter_to_continue()
            clear_screen()
        # Display the instructions and wait before returning to the main menu
        elif choice == 2:
            read_instructions()
            press_enter_to_continue()
            clear_screen()
        # Display the credits and wait before returning to the main menu
        elif choice == 3:
            show_credits()
            press_enter_to_continue()
            clear_screen()
        # Ask for confirmation before closing the application
        elif choice == 4:
            if confirm_exit():
                print("\nThank you for using Y-MAX. Goodbye!")
                print("Exiting the program...")
                break
            print("\nReturning to the main menu...")



if __name__ == "__main__":  # Run the main function only when this file is executed directly
    main()
