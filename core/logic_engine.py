import time
from multiprocessing import Queue
from collections import defaultdict
import numpy as np
import cv2
import os
import config
from bytetrack.bytetrack_simple import SimpleBYTETracker
from src.face_recognition.app.detector import detect_faces
from src.face_recognition.app.embedder import get_embedding
class FaceRecognizer:
    def __init__(self, data_dir="data"):
        """
        Initializes the face recognizer by loading embeddings and names from .npy files.
        """
        print("[Face Recognizer] Initializing...")
        self.known_embeddings = None
        self.known_names = None
        embeddings_file = config.FACE_EMBEDDINGS_PATH
        names_file = config.FACE_NAMES_PATH

        if os.path.exists(embeddings_file) and os.path.exists(names_file):
            try:
                self.known_embeddings = np.load(embeddings_file)
                self.known_names = np.load(names_file, allow_pickle=True)

                if self.known_embeddings.shape[0] == self.known_names.shape[0]:
                    print(f"[Face Recognizer] ✅ Loaded {len(self.known_names)} face embeddings and names.")
                else:
                    print("[Face Recognizer] 🔴 ERROR: Mismatch between embeddings and names file sizes.")
                    self.known_embeddings = None
                    self.known_names = None
            except Exception as e:
                print(f"[Face Recognizer] 🔴 ERROR loading files: {e}")
        else:
            print(f"[Face Recognizer] 🔴 ERROR: One or more files not found.")
            print(f"  - Searched for embeddings at: {embeddings_file}")
            print(f"  - Searched for names at: {names_file}")

    def _cosine_similarity(self, a, b):
        """Helper function to calculate cosine similarity."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _find_match(self, embedding_to_check):
        """Finds the best match for a given embedding using cosine similarity."""
        best_match_name = "Unknown"
        highest_similarity = 0.0
        potential_name = "Unknown"

        for i, known_emb in enumerate(self.known_embeddings):
            similarity = self._cosine_similarity(embedding_to_check, known_emb)
            if similarity > highest_similarity:
                highest_similarity = similarity
                potential_name = self.known_names[i]

        print(
            f"[Face Recognizer] [DEBUG] Best similarity: {highest_similarity:.2f} | Threshold: {config.FACE_RECOGNITION_THRESHOLD}")

        if highest_similarity > config.FACE_RECOGNITION_THRESHOLD:
            best_match_name = potential_name

        return best_match_name

    def recognize(self, person_crop_image):
        """
        Detects a face and finds the closest match from known embeddings.
        """
        if self.known_embeddings is None or person_crop_image is None or person_crop_image.size == 0:
            return "Unknown"

        try:
            faces = detect_faces(person_crop_image)
            if faces:
                face_image, _ = faces[0]
                embedding = get_embedding(face_image)
                name = self._find_match(embedding)
                return name
        except Exception:
            pass

        return "Unknown"
def check_overlap(person_bbox, item_bbox):
    px1, py1, px2, py2 = person_bbox
    ix1, iy1, ix2, iy2 = item_bbox
    return not (px2 < ix1 or px1 > ix2 or py2 < iy1 or py1 > iy2)
def process_logic(results_queue: Queue, alert_queue: Queue):
    print("[Logic Engine] 🟢 Starting...")

    trackers = {}
    tracked_person_states = defaultdict(lambda: {
        "name": "Unknown",
        "last_alert_times": defaultdict(float),
        "violation_confirm_counter": 0,
        "current_violations": set()
    })

    face_recognizer = FaceRecognizer()
    REQUIRED_PPE = {"helmet", "vest"}

    while True:
        try:
            data = results_queue.get(timeout=1)
            camera_id, original_frame, all_detections = data["camera_id"], data["original_frame"], data["detections"]

            all_class_names = [d['class_name'] for d in all_detections]
            print(f"[Logic Engine] [DEBUG] Cam {camera_id} received detections: {all_class_names}")

            if camera_id not in trackers:
                trackers[camera_id] = SimpleBYTETracker(
                    track_thresh=config.TRACK_THRESH,
                    track_buffer=config.TRACK_BUFFER,
                    match_thresh=config.MATCH_THRESH
                )

            person_dets_track, ppe_items, env_alerts = [], [], []
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

                detected_ppe_for_person, explicit_violations = set(), set()
                for item in ppe_items:
                    if check_overlap(person_bbox, item["bbox"]):
                        if item["class_name"].startswith("no-"):
                            explicit_violations.add(item["class_name"])
                        else:
                            detected_ppe_for_person.add(item["class_name"])

                missing_ppe = REQUIRED_PPE - detected_ppe_for_person
                violations_this_frame = explicit_violations.union({f"missing-{item}" for item in missing_ppe})

                if violations_this_frame:
                    state["violation_confirm_counter"] = min(config.VIOLATION_CONFIRM_FRAMES,
                                                             state["violation_confirm_counter"] + 1)
                    state["current_violations"] = violations_this_frame
                else:
                    state["violation_confirm_counter"] = max(0, state["violation_confirm_counter"] - 1)
                    if state["violation_confirm_counter"] == 0:
                        state["current_violations"].clear()

                if state["violation_confirm_counter"] >= config.VIOLATION_CONFIRM_FRAMES:
                    current_time = time.time()
                    new_alerts_to_send = []
                    for violation in state["current_violations"]:
                        if (current_time - state["last_alert_times"][violation]) > config.ALERT_COOLDOWN_SECONDS:
                            new_alerts_to_send.append(violation)
                            state["last_alert_times"][violation] = current_time

                    if new_alerts_to_send:
                        alert = {"type": "ppe_violation", "camera_id": camera_id, "person_name": state["name"],
                                 "track_id": track_id, "violations": new_alerts_to_send}
                        alert_queue.put(alert)

                    state["violation_confirm_counter"] = 0

            for alert in env_alerts:
                alert_data = {"type": "environmental_alert", "camera_id": camera_id, "alert_type": alert["class_name"],
                              "bbox": alert["bbox"].tolist()}
                alert_queue.put(alert_data)
        except Exception:
            pass
