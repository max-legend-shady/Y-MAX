# Import the libraries required for timing, file handling, model loading, and media processing
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from ultralytics import YOLO
import math
import os
import time
import cv2

# Define the project directory and all input, output, and model folders
project_directory = Path(__file__).resolve().parent
input_images = project_directory / "input" / "images"
output_images = project_directory / "output" / "images"
input_videos = project_directory / "input" / "videos"
output_videos = project_directory / "output" / "videos"
models_folder = project_directory / "models"
# Create the folders required by Y-MAX if they do not already exist
for folder in (input_images, output_images, input_videos, output_videos, models_folder):
    folder.mkdir(parents=True, exist_ok=True)
# Define the supported file extensions for images, videos, and custom models
image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
model_extensions = [".pt"]
# Define the available pretrained YOLO models in the same order as the model menu
model_files = ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"]
# Define the project version and every default value used by an input or processing fallback
project_version = "1.1.0"
default_model = "yolo26n.pt"
default_threshold = 0.5
default_image_folder = input_images
default_video_folder = input_videos
default_video_fps = 30.0
# Define the YOLO class index used to detect people
class_index = 0


def introduction():
    # Display the project name, version, and author
    print("\n--- WELCOME TO Y-MAX ---")
    print("YOLO-based Multimodal Analysis eXplorer")
    print(f"Version {project_version}")
    print("\nCreated by Manuel Heredia")


def make_menu(title, options):
    # Display the menu title and add the project heading on the main menu
    if title == "MAIN MENU":
        print("--- Y-MAX ---")
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
    input("\nEnter to continue: ").lower().strip()


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


def print_batch_summary(media_label, batch_start, selected_count, completed_count, failed_count, interrupted_count, total_detections, total_frames, input_bytes, output_bytes, processing_seconds):
    # Calculate the total elapsed time, including model loading and processing
    total_elapsed = time.perf_counter() - batch_start
    print(f"\n--- {media_label.upper()} BATCH SUMMARY ---")
    # Report how many selected files completed, failed, or were interrupted
    print(f"Files selected: {selected_count}")
    print(f"Files completed: {completed_count}")
    print(f"Files failed: {failed_count}")
    print(f"Files interrupted: {interrupted_count}")
    # Report the total detection count and data sizes
    print(f"Total person detections: {total_detections}")
    print(f"Input data processed: {format_bytes(input_bytes)}")
    print(f"Output data saved: {format_bytes(output_bytes)}")
    # Report the total task duration and average processing rate
    print(f"Total task time: {total_elapsed:.2f} seconds")
    print(f"Processing time: {processing_seconds:.2f} seconds")
    if media_label == "video" and processing_seconds > 0:
        print(f"Average processing FPS: {total_frames / processing_seconds:.2f}")
        print(f"Total frames processed: {total_frames}")
    elif media_label == "image" and processing_seconds > 0:
        print(f"Average image processing rate: {(completed_count + failed_count) / processing_seconds:.2f} images/second")


def get_pretrained_model_path(model_filename):
    # Prefer a pretrained model file stored in the project's models folder
    local_model_path = models_folder / model_filename
    if local_model_path.is_file():
        return str(local_model_path)
    # Let Ultralytics locate or download the model when it is not stored locally
    return model_filename


def initiate_detection():
    # Keep the media menu available when the user backs out of model selection
    while True:
        choice = make_menu("MEDIA TYPE SELECTION", ["Image(s)", "Video(s)", "Back"])
        # Start image analysis and show its result screen when the workflow finishes
        if choice == 1:
            print("Preparing image analysis...")
            if run_image_analysis():
                return True
            continue
        # Start video analysis and show its result screen when the workflow finishes
        if choice == 2:
            print("Preparing video analysis...")
            if run_video_analysis():
                return True
            continue
        # Return to the main menu when the user selects Back from media selection
        print("Returning to the main menu...")
        return False


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
    elif file_type == "video":
        default_folder = default_video_folder
        extensions = video_extensions
        file_label = "video"
        plural_label = "videos"
    else:
        # Prevent unsupported internal file types from being processed
        raise ValueError("Unsupported file type. Use 'image' or 'video'.")
    # Return to the previous workflow menu when the user backs out of folder selection
    while True:
        folder_path_text = input(f"Enter the folder path (leave blank for input/{plural_label}, or enter 'back'): ").strip().strip("&").strip().strip('"').strip("'")
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
            selection = input(f"\nEnter the {file_label} numbers separated by commas, 'all' to select every {file_label}, or 'back': ").lower().strip()
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


def choose_confidence_threshold():
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


def run_image_detection(model, model_name, image_path, confidence_threshold):
    # Run YOLO detection on the image and filter results to the person class
    results = model.predict(source=str(image_path), classes=[class_index], conf=confidence_threshold, verbose=False)
    # Retrieve the result for the single selected image
    result = results[0]
    # Count the detected people in the image
    detected_people = len(result.boxes)
    # Create an annotated image containing the detection boxes and labels
    annotated_image = result.plot()
    # Build an output filename containing the input name, model, and confidence threshold
    model_label = Path(model_name).stem
    output_name = f"{image_path.stem}_annotated_model={model_label}_confidence={confidence_threshold}"
    output_path = output_images / f"{output_name}.jpg"
    # Add a number to the output filename if a previous result already exists
    counter = 2
    while output_path.exists():
        output_path = output_images / f"{output_name}_{counter}.jpg"
        counter += 1
    # Save the annotated image and report an error if saving fails
    if not cv2.imwrite(str(output_path), annotated_image):
        raise OSError("Could not save the annotated image.")
    # Return the detection count and saved output path
    return detected_people, output_path


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
    # Build a unique output filename using the source video, model, and confidence threshold
    model_label = Path(model_name).stem
    output_name = f"{video_path.stem}_annotated_model={model_label}_confidence={confidence_threshold}"
    output_path = output_videos / f"{output_name}.mp4"
    # Avoid overwriting an existing annotated video
    counter = 2
    while output_path.exists():
        output_path = output_videos / f"{output_name}_{counter}.mp4"
        counter += 1
    # Create the video writer using the MP4V codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, (frame_width, frame_height))
    # Check that OpenCV successfully created the output video
    if not writer.isOpened():
        capture.release()
        writer.release()
        raise OSError("Could not create the output video.")
    # Initialise counters for processed frames and detections across all frames
    frame_count = 0
    total_detections = 0
    try:
        # Read, analyse, annotate, and write each frame until the video ends or the user interrupts
        while True:
            success, frame = capture.read()
            # Stop when OpenCV reaches the end of the video
            if not success:
                break
            # Run person detection on the current frame
            results = model.predict(source=frame, classes=[class_index], conf=confidence_threshold, verbose=False)
            result = results[0]
            # Add this frame's detections to the running total
            total_detections += len(result.boxes)
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
        return False, frame_count, total_detections, None, elapsed, output_fps
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
    return True, frame_count, total_detections, output_path, elapsed, output_fps


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
                    finished, frame_count, detections, output_path, file_elapsed, output_fps = run_video_detection(model, model_name, video_path, confidence_threshold)
                    processing_seconds += file_elapsed
                    total_frames += frame_count
                    total_detections += detections
                    # Stop the remaining queue if the current video was interrupted
                    if not finished:
                        interrupted_count += 1
                        print(f"Frames processed before interruption: {frame_count}")
                        print(f"Detections before interruption: {detections}")
                        print(f"Time elapsed before interruption: {file_elapsed:.2f} seconds")
                        print(f"Average processing FPS before interruption: {frame_count / file_elapsed:.2f}" if file_elapsed > 0 else "Average processing FPS: unavailable")
                        print(f"Input size: {format_bytes(input_size)}")
                        print("Output size: 0 B (incomplete video removed)")
                        break
                    # Record the completed video and its output size
                    output_size = get_file_size(output_path)
                    output_bytes += output_size
                    completed_count += 1
                    # Display detailed statistics for the completed video
                    print(f"Processed frames: {frame_count}")
                    print(f"Total detections across frames: {detections}")
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
            # Display the final batch report and output location
            print_batch_summary("video", batch_start, len(selected_videos), completed_count, failed_count, interrupted_count, total_detections, total_frames, input_bytes, output_bytes, processing_seconds)
            print(f"Annotated videos are saved in: {output_videos}")
            return True


def read_instructions():
    # Display setup instructions and the complete image/video analysis workflow
    print("\n--- INSTRUCTIONS ---")
    print("\n--- STEP 1: PREPARATION ---")
    print("Install Python 3.13 or newer and uv, then run 'uv sync' from the Y-MAX project folder.")
    print("Y-MAX creates the required input, output, and models folders automatically.")
    print("Place images in 'input/images' and videos in 'input/videos', or select another folder when prompted.")
    print("\n--- STEP 2: NAVIGATION AND FILE SELECTION ---")
    print("Choose Image(s) or Video(s) from the media type menu.")
    print("At folder selection, press Enter to use the relevant default input folder.")
    print("Y-MAX searches the selected folder and all its subfolders for supported media files.")
    print("Select individual files by entering their numbers separated by commas, or enter 'all' to select every file found.")
    print("Type 'back' at any selection prompt to return to the immediately previous menu or selection step.")
    print("From the file list, 'back' returns to folder selection; from folder selection, it returns to confidence selection.")
    print("From confidence selection, 'back' returns to model selection; from model selection, it returns to media selection.")
    print("\n--- STEP 3: MODEL AND CONFIDENCE ---")
    print("Choose YOLO26 Nano, Small, Medium, Large, Extra Large, or a custom '.pt' model.")
    print(f"When entering a custom model path, press Enter to use the default model ({default_model}).")
    print("Pretrained weights are loaded from the 'models' folder when present; otherwise Ultralytics may download them.")
    print(f"Enter a confidence threshold between 0 and 1. Press Enter to use the default value {default_threshold}.")
    print("Only detections with confidence equal to or greater than the selected threshold are included.")
    print("\n--- STEP 4: DETECTION AND CANCELLATION ---")
    print("Y-MAX currently detects people only, using YOLO class index 0.")
    print("Images are processed one at a time. Videos are processed one frame at a time.")
    print("Press Ctrl+C at any time during processing to stop the current batch.")
    print("All outputs completed before the interruption remain saved.")
    print("If an image is interrupted before saving, that image is not saved. An interrupted video's incomplete output is removed.")
    print("Ctrl+C stops the batch; it does not pause and resume the current file.")
    print("\n--- STEP 5: RESULTS AND STATISTICS ---")
    print("Annotated images are saved in 'output/images'. Annotated videos are saved in 'output/videos'.")
    print("Output filenames include the source name, selected model, and confidence threshold; existing files are not overwritten.")
    print("Per-file and batch reports include elapsed time, input/output sizes, detections, and completion counts.")
    print("Video reports also include processed frames, output FPS, and average processing FPS.")
    print("Video detection totals count detections across frames, not unique people; the same person can be counted repeatedly.")
    print("\n--- SUPPORTED FORMATS AND LIMITATIONS ---")
    print(f"Image formats: {', '.join(image_extensions)}.")
    print(f"Video formats: {', '.join(video_extensions)}.")
    print(f"Custom model formats: {', '.join(model_extensions)}.")
    print("OpenCV-generated output videos do not preserve the source video's audio.")
    print("Y-MAX assumes class index 0 represents people in the selected model.")
    print("Detection may produce false positives or false negatives; manually check important results.")
    print("\nThank you for using Y-MAX!")


def show_credits():
    # Display the project name, current version, and author
    print("\n--- CREDITS ---")
    print("Y-MAX: YOLO-based Multimodal Analysis eXplorer")
    print(f"Version {project_version}")
    print("Program written by Manuel Heredia")
    # Display versions of the direct dependencies used by this project
    print("\nDirect dependencies:")
    for package_name in ("ultralytics", "opencv-python", "numpy"):
        try:
            # Retrieve the installed version of the current dependency
            package_version = version(package_name)
            print(f"- {package_name} ({package_version})")
        except PackageNotFoundError:
            # Report a dependency that is not installed in the active environment
            print(f"- {package_name} (not installed)")
    # Describe the educational purpose of the project
    print("\nThis program was made as a personal project to learn more about computer vision and object detection.")
    print("I am also going to write an article on this project, so stay tuned!")
    print("If you have any questions or feedback, please feel free to reach out to me at the following email: manu.heredia.estrada@gmail.com")
    print("\nThank you for using Y-MAX!")


def confirm_exit():
    # Keep asking until the user confirms, declines, or cancels the exit
    while True:
        confirmation = input("Enter 'yes' to exit, leave blank to return to the main menu: ").lower().strip()
        # Confirm the exit when the user enters 'y' or 'yes'
        if confirmation in ("yes", "y"):
            return True
        # Return to the main menu when the user does not enter 'y' or 'yes'
        return False


def main():
    # Clear the terminal and display the introduction
    clear_screen()
    introduction()
    # Keep displaying the main menu until the user confirms exit
    while True:
        choice = make_menu("MAIN MENU", ["Detect people in an image or video", "Read the instructions", "Show the credits", "Exit"])
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


# Run the main function only when this file is executed directly
if __name__ == "__main__":
    main()
