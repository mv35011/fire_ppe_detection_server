# --- Video Input Settings ---
# A dictionary mapping a unique camera ID to its video source path.
# Use 0, 1, 2, etc., for camera IDs.
# The path can be a local video file or a network stream URL (e.g., rtsp://...).
CAMERA_FEEDS = {
    0: "path/to/your/dummy_video_1.mp4",
    1: "path/to/your/dummy_video_2.mp4",
    2: "path/to/your/dummy_video_3.mp4",
    # Add more camera feeds as needed
}

# The target frames per second for the input handlers.
TARGET_FPS = 10

# --- Queue Settings ---
# The maximum number of frames to hold in memory before the input handlers
# start skipping frames. Helps prevent memory overload.
FRAME_QUEUE_SIZE = 50

# --- AI Model Settings ---
# Path to your trained YOLO model weights.
# This should be your unified model for persons, PPE, fire, and smoke.
MODEL_PATH = "models/unified_model.pt"

# The number of frames to batch together for inference.
# Adjust based on your GPU's VRAM. A higher batch size is more efficient.
# For a Tesla T4 or similar, 4 or 8 is a good starting point.
INFERENCE_BATCH_SIZE = 4

# The minimum confidence score for a detection to be considered valid.
CONF_THRESHOLD = 0.3

# The IoU threshold for non-maximum suppression (removes overlapping boxes).
IOU_THRESHOLD = 0.4
