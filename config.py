# --- Video Input Settings ---
# A dictionary mapping a unique camera ID to its video source path.
# Use 0, 1, 2, etc., for camera IDs.
# The path can be a local video file or a network stream URL (e.g., rtsp://...).
CAMERA_FEEDS = {
    0: r"D:\Compressed\ppe_fire_final_server\videos\20250809_102632.mp4",
    1: r"D:\Compressed\ppe_fire_final_server\videos\Helmet Googles Jacket No Boots No Gloves.mp4",
    2: r"D:\Compressed\ppe_fire_final_server\videos\fire.mp4"

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

PPE_MODEL_PATH="models/ppe_detection.pt"
FIRE_MODEL_PATH="models/best_fire_40epochs.pt"
# The number of frames to batch together for inference.
# Adjust based on your GPU's VRAM. A higher batch size is more efficient.
# For a Tesla T4 or similar, 4 or 8 is a good starting point.
INFERENCE_BATCH_SIZE = 4

PERSON_MODEL_PATH = "yolov8n.pt"

# The minimum confidence score for a detection to be considered valid.
CONF_THRESHOLD = 0.1

# The IoU threshold for non-maximum suppression (removes overlapping boxes).
IOU_THRESHOLD = 0.2
