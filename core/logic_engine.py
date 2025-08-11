import time
from multiprocessing import Queue
from collections import defaultdict
import numpy as np
import cv2

# --- Project-specific Imports ---
import config
from bytetrack.bytetrack_simple import SimpleBYTETracker

# --- Face Recognition Imports (from your colleague's code) ---
# Ensure these modules are accessible from your project's path.
# You will need to install deepface and psycopg2:
# pip install deepface psycopg2-binary
from src.face_recognition.app.database import fetch_all_embeddings
from src.face_recognition.app.detector import detect_faces
from src.face_recognition.app.embedder import get_embedding
from src.face_recognition.app.matcher import find_match


# ==============================================================================
# ## Integrated Face Recognition Module ##
# This class now uses the actual face recognition functions.
# ==============================================================================
class FaceRecognizer:
    def __init__(self):
        """
        Loads the known face embeddings from the database upon initialization.
        """
        print("[Face Recognizer] Initializing...")
        self.known_embeddings = []
        try:
            # Fetch all names and embeddings from the PostgreSQL database
            self.known_embeddings = fetch_all_embeddings()
            if self.known_embeddings:
                print(
                    f"[Face Recognizer] ✅ Successfully loaded {len(self.known_embeddings)} known faces from the database.")
            else:
                print("[Face Recognizer] 🟡 WARNING: No face embeddings found in the database.")
        except Exception as e:
            print(f"[Face Recognizer] 🔴 ERROR: Could not connect to or fetch from the database: {e}")
            print("[Face Recognizer] 🔴 Face recognition will not be available.")

    def recognize(self, person_crop_image):
        """
        Recognizes a face from a cropped image of a person.

        Args:
            person_crop_image: A small image containing the tracked person.

        Returns:
            The name of the recognized person or "Unknown".
        """
        if not self.known_embeddings:
            return "Unknown"  # Cannot perform recognition if DB is empty

        try:
            # 1. Detect faces within the person crop using RetinaFace
            # The `detect_faces` function returns a list of (face_image, facial_area)
            detected_faces_list = detect_faces(person_crop_image)

            # If a face is found in the crop
            if detected_faces_list:
                # Use the first detected face for simplicity
                face_image, _ = detected_faces_list[0]

                # 2. Get the embedding for the detected face using ArcFace
                embedding = get_embedding(face_image)

                # 3. Match the embedding against the database
                is_match, name = find_match(embedding, self.known_embeddings)

                if is_match:
                    return name

        except Exception as e:
            # This can happen if DeepFace fails on a low-quality crop
            # print(f"[Face Recognizer] Could not process face: {e}")
            pass

        return "Unknown"


# --- Helper Function ---
def check_overlap(person_bbox, item_bbox):
    """A simple helper to check if two bounding boxes overlap."""
    px1, py1, px2, py2 = person_bbox
    ix1, iy1, ix2, iy2 = item_bbox
    return not (px2 < ix1 or px1 > ix2 or py2 < iy1 or py1 > iy2)


# --- Main Logic Function ---
def process_logic(results_queue: Queue, alert_queue: Queue):
    """
    A target function for the logic and tracking process.
    """
    print("[Logic Engine] 🟢 Starting...")

    trackers = {}
    tracked_person_states = defaultdict(lambda: {
        "name": "Unknown",
        "last_alert_time": 0,
        "violation_confirm_counter": 0,
        "current_violations": set()
    })

    # Initialize the fully functional face recognizer
    face_recognizer = FaceRecognizer()

    while True:
        try:
            data = results_queue.get(timeout=1)
            camera_id = data["camera_id"]
            original_frame = data["original_frame"]
            all_detections = data["detections"]

            if camera_id not in trackers:
                print(f"[Logic Engine] Initializing new tracker for Camera {camera_id}")
                trackers[camera_id] = SimpleBYTETracker(track_thresh=config.CONF_THRESHOLD)

            person_detections_for_tracking = []
            ppe_violations = []
            fire_smoke_alerts = []

            for det in all_detections:
                if det["class_name"] == "person":
                    person_detections_for_tracking.append(det["bbox"].tolist() + [det["score"]])
                elif det["class_name"].startswith("no-"):
                    ppe_violations.append(det)
                elif det["class_name"] in ["fire", "smoke"]:
                    fire_smoke_alerts.append(det)

            tracked_persons = trackers[camera_id].update(np.array(person_detections_for_tracking),
                                                         original_frame.shape[:2])

            for person in tracked_persons:
                track_id = person.track_id
                person_bbox = person.bbox
                state = tracked_person_states[track_id]

                if state["name"] == "Unknown":
                    x1, y1, x2, y2 = map(int, person_bbox)
                    # Ensure crop coordinates are valid
                    if x1 < x2 and y1 < y2:
                        person_crop = original_frame[y1:y2, x1:x2]

                        # Use the integrated face recognizer
                        name = face_recognizer.recognize(person_crop)
                        if name != "Unknown":
                            state["name"] = name
                            print(f"[Logic Engine] Identified Track ID {track_id} as '{name}'")

                violations_this_frame = set()
                for violation in ppe_violations:
                    if check_overlap(person_bbox, violation["bbox"]):
                        violations_this_frame.add(violation["class_name"])

                if violations_this_frame:
                    state["violation_confirm_counter"] += 1
                    state["current_violations"].update(violations_this_frame)
                else:
                    state["violation_confirm_counter"] = 0
                    state["current_violations"].clear()

                if state["violation_confirm_counter"] >= config.VIOLATION_CONFIRM_FRAMES:
                    current_time = time.time()
                    if (current_time - state["last_alert_time"]) > config.ALERT_COOLDOWN_SECONDS:
                        alert_queue.put({
                            "type": "ppe_violation",
                            "camera_id": camera_id,
                            "person_name": state["name"],
                            "track_id": track_id,
                            "violations": list(state["current_violations"]),
                            "bbox": person_bbox.tolist()
                        })
                        state["last_alert_time"] = current_time
                        state["violation_confirm_counter"] = 0

            for alert in fire_smoke_alerts:
                alert_queue.put({
                    "type": "environmental_alert",
                    "camera_id": camera_id,
                    "alert_type": alert["class_name"],
                    "bbox": alert["bbox"].tolist()
                })

        except Exception:
            pass
