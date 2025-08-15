# --- Video Input Settings ---
CAMERA_FEEDS = {
    0: r"videos\20250809_102632 (1).mp4",
    1: r"videos\fire.mp4",
    2: r"videos\Helmet Googles Jacket No Boots No Gloves (1).mp4"
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
# Paths to your .npy files
FACE_EMBEDDINGS_PATH = "data/embeddings.npy"
# ❗ UPDATED: Added a specific path for your names file.
FACE_NAMES_PATH = "data/n.npy"

# The similarity score required to consider a face a match.
# ❗ UPDATED: Lowered threshold to a more reasonable value for ArcFace.
FACE_RECOGNITION_THRESHOLD = 0.7

# --- Logic Engine Settings ---
VIOLATION_CONFIRM_FRAMES = 3
ALERT_COOLDOWN_SECONDS = 10

# --- ByteTrack Settings ---
TRACK_THRESH = 0.5
TRACK_BUFFER = 60
MATCH_THRESH = 0.8
