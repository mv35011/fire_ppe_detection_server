# --- Video Input Settings ---
CAMERA_FEEDS = {
    0: r"D:\Compressed\ppe_fire_final_server\videos\20250809_102632.mp4",
    1: r"D:\Compressed\ppe_fire_final_server\videos\Helmet Googles Jacket No Boots No Gloves.mp4",
    2: r"D:\Compressed\ppe_fire_final_server\videos\fire.mp4"
}
TARGET_FPS = 10

# --- Queue Settings ---
FRAME_QUEUE_SIZE = 50

# --- AI Model Settings ---
PERSON_MODEL_PATH = "yolov8n.pt"
PPE_MODEL_PATH = "models/ppe_detection.pt"
FIRE_MODEL_PATH = "models/best_fire_40epochs.pt"
INFERENCE_BATCH_SIZE = 4
CONF_THRESHOLD = 0.1
IOU_THRESHOLD = 0.2

# --- Face Recognition Settings ---
# This should point to your .npy file with face embeddings
FACE_EMBEDDINGS_PATH = "data/embeddings.npy"

# --- Logic Engine Settings ---
# Number of consecutive frames a violation must be detected before an alert is triggered.
# Lower this to make alerts more frequent for testing.
VIOLATION_CONFIRM_FRAMES = 3

# Cooldown in seconds for each specific violation type per person.
ALERT_COOLDOWN_SECONDS = 10

# --- ByteTrack Settings ---
# High-confidence detection threshold for matching
TRACK_THRESH = 0.5

# How many frames to keep a track alive after it's lost.
# Increasing this can help with unstable tracking.
TRACK_BUFFER = 60

# IoU threshold for matching detections to existing tracks.
MATCH_THRESH = 0.8
