import time
from multiprocessing import Queue
from collections import defaultdict
import numpy as np
import cv2

# --- Project-specific Imports ---
import config
from bytetrack.bytetrack_simple import SimpleBYTETracker

# --- Face Recognition Imports ---
# (Assuming these are correctly set up as before)
from src.face_recognition.app.database import fetch_all_embeddings
from src.face_recognition.app.detector import detect_faces
from src.face_recognition.app.embedder import get_embedding
from src.face_recognition.app.matcher import find_match


# ==============================================================================
# ## Integrated Face Recognition Module (No changes needed here) ##
# ==============================================================================
class FaceRecognizer:
    def __init__(self):
        print("[Face Recognizer] Initializing...")
        self.known_embeddings = []
        try:
            self.known_embeddings = fetch_all_embeddings()
            if self.known_embeddings:
                print(f"[Face Recognizer] ✅ Loaded {len(self.known_embeddings)} faces.")
            else:
                print("[Face Recognizer] 🟡 WARNING: No face embeddings in database.")
        except Exception as e:
            print(f"[Face Recognizer] 🔴 ERROR connecting to database: {e}")

    def recognize(self, person_crop_image):
        if not self.known_embeddings: return "Unknown"
        try:
            faces = detect_faces(person_crop_image)
            if faces:
                face_image, _ = faces[0]
                embedding = get_embedding(face_image)
                is_match, name = find_match(embedding, self.known_embeddings)
                if is_match:
                    return name
        except Exception:
            pass
        return "Unknown"


# --- Helper Function ---
def check_overlap(person_bbox, item_bbox):
    px1, py1, px2, py2 = person_bbox
    ix1, iy1, ix2, iy2 = item_bbox
    return not (px2 < ix1 or px1 > ix2 or py2 < iy1 or py1 > iy2)


# --- Main Logic Function ---
def process_logic(results_queue: Queue, alert_queue: Queue):
    print("[Logic Engine] 🟢 Starting...")

    trackers = {}
    tracked_person_states = defaultdict(lambda: {
        "name": "Unknown", "last_alert_time": 0,
        "violation_confirm_counter": 0, "current_violations": set()
    })

    face_recognizer = FaceRecognizer()

    REQUIRED_PPE = {"helmet", "vest"}

    while True:
        try:
            data = results_queue.get(timeout=1)

            camera_id, original_frame, all_detections = data["camera_id"], data["original_frame"], data["detections"]

            # --- DEBUG: Print all received class names ---
            # This will show you the exact names your models are outputting.
            all_class_names = [d['class_name'] for d in all_detections]
            print(f"[Logic Engine] [DEBUG] Cam {camera_id} received detections: {all_class_names}")

            if camera_id not in trackers:
                trackers[camera_id] = SimpleBYTETracker(track_thresh=config.CONF_THRESHOLD)

            person_dets_track = []
            ppe_items = []
            env_alerts = []
            for det in all_detections:
                if det["class_name"] == "person":
                    person_dets_track.append(det["bbox"].tolist() + [det["score"]])
                elif det["class_name"] in ["fire", "smoke"]:
                    env_alerts.append(det)
                else:
                    ppe_items.append(det)

            tracked_persons = trackers[camera_id].update(np.array(person_dets_track), original_frame.shape[:2])

            for person in tracked_persons:
                track_id, person_bbox = person.track_id, person.bbox
                state = tracked_person_states[track_id]

                if state["name"] == "Unknown":
                    x1, y1, x2, y2 = map(int, person_bbox)
                    if x1 < x2 and y1 < y2:
                        name = face_recognizer.recognize(original_frame[y1:y2, x1:x2])
                        if name != "Unknown":
                            state["name"] = name
                            print(f"[Logic Engine] Identified Track ID {track_id} as '{name}'")

                detected_ppe_for_person = set()
                explicit_violations = set()
                for item in ppe_items:
                    if check_overlap(person_bbox, item["bbox"]):
                        if item["class_name"].startswith("no-"):
                            explicit_violations.add(item["class_name"])
                        else:
                            detected_ppe_for_person.add(item["class_name"])

                missing_ppe = REQUIRED_PPE - detected_ppe_for_person
                violations_this_frame = explicit_violations.union({f"missing-{item}" for item in missing_ppe})

                if violations_this_frame:
                    state["violation_confirm_counter"] += 1
                    state["current_violations"].update(violations_this_frame)
                else:
                    state["violation_confirm_counter"] = 0
                    state["current_violations"].clear()

                if state["violation_confirm_counter"] >= config.VIOLATION_CONFIRM_FRAMES:
                    current_time = time.time()
                    if (current_time - state["last_alert_time"]) > config.ALERT_COOLDOWN_SECONDS:
                        alert = {
                            "type": "ppe_violation", "camera_id": camera_id,
                            "person_name": state["name"], "track_id": track_id,
                            "violations": list(state["current_violations"]),
                        }
                        alert_queue.put(alert)
                        state["last_alert_time"] = current_time
                        state["violation_confirm_counter"] = 0

            for alert in env_alerts:
                alert_data = {
                    "type": "environmental_alert", "camera_id": camera_id,
                    "alert_type": alert["class_name"], "bbox": alert["bbox"].tolist()
                }
                alert_queue.put(alert_data)

        except Exception:
            pass
