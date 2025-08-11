import time
from multiprocessing import Queue
from collections import defaultdict

import config
# Assuming your bytetrack script is in a place Python can find it
# e.g., in a 'bytetrack' directory with an __init__.py
from bytetrack.bytetrack_simple import SimpleBYTETracker


# --- Helper Function ---
def check_overlap(person_bbox, item_bbox):
    """Checks if two bounding boxes overlap."""
    px1, py1, px2, py2 = person_bbox
    ix1, iy1, ix2, iy2 = item_bbox
    return not (px2 < ix1 or px1 > ix2 or py2 < iy1 or py1 > iy2)


# --- Main Logic Function ---
def process_logic(results_queue: Queue, alert_queue: Queue):
    """
    A target function for the logic and tracking process.

    This function consumes detection results, applies tracking, manages state,
    and generates alerts for safety violations.

    Args:
        results_queue (Queue): The queue to get raw detection results from.
        alert_queue (Queue): The queue to send final, confirmed alerts to.
    """
    print("[Logic Engine] 🟢 Starting...")

    # A dictionary to hold a separate tracker for each camera feed
    trackers = {}

    # A dictionary to hold the state of each tracked person across all cameras.
    # The key will be the track_id.
    tracked_person_states = defaultdict(lambda: {
        "name": "Unknown",
        "last_alert_time": 0,
        "violation_confirm_counter": 0,
        "current_violations": set()
    })

    # A dictionary to map class IDs from the model to human-readable names
    # This should match the training configuration of your unified model.
    CLASS_NAMES = {
        0: "person", 1: "helmet", 2: "no-helmet", 3: "vest", 4: "no-vest",
        5: "fire", 6: "smoke"
        # ... add all your classes here
    }

    while True:
        try:
            # Get the next result from the inference engine
            data = results_queue.get()
            camera_id = data["camera_id"]
            original_frame = data["original_frame"]
            detections = data["detections"]

            # 1. Initialize a tracker if it's the first time we see this camera
            if camera_id not in trackers:
                print(f"[Logic Engine] Initializing new tracker for Camera {camera_id}")
                trackers[camera_id] = SimpleBYTETracker(
                    track_thresh=config.CONF_THRESHOLD,
                    match_thresh=0.8
                )

            # 2. Separate detections for tracking
            person_detections = []
            other_detections = []
            for det in detections:
                class_name = CLASS_NAMES.get(det["class_id"], "unknown")
                if class_name == "person":
                    # ByteTrack expects [x1, y1, x2, y2, score]
                    person_detections.append(det["bbox"].tolist() + [det["score"]])
                else:
                    other_detections.append({"bbox": det["bbox"], "class_name": class_name})

            # 3. Update the tracker for the current camera feed
            tracked_persons = trackers[camera_id].update(person_detections, original_frame.shape[:2])

            # 4. Process each tracked person
            for person in tracked_persons:
                track_id = person.track_id
                person_bbox = person.bbox
                state = tracked_person_states[track_id]

                # --- Face Recognition Logic ---
                if state["name"] == "Unknown":
                    # ## PLACEHOLDER: FACE RECOGNITION ##
                    # Your colleague's work goes here.
                    # 1. Crop the face from the original_frame using the person_bbox
                    #    x1, y1, x2, y2 = map(int, person_bbox)
                    #    person_crop = original_frame[y1:y2, x1:x2]
                    #
                    # 2. Run RetinaFace to detect the face within the crop.
                    #    face_image = detect_face_with_retinaface(person_crop)
                    #
                    # 3. If a face is found, get its embedding.
                    #    embedding = get_face_embedding(face_image)
                    #
                    # 4. Match the embedding against your database.
                    #    matched_name = find_match_in_db(embedding)
                    #
                    # 5. If a match is found, update the state.
                    #    if matched_name:
                    #        state["name"] = matched_name
                    #        print(f"[Logic Engine] Identified Track ID {track_id} as {matched_name}")
                    pass  # Keep as "Unknown" for now

                # --- Violation Checking Logic ---
                violations_this_frame = set()
                for item in other_detections:
                    # Check if a violation item (e.g., 'no-helmet', 'fire') overlaps with the person
                    if check_overlap(person_bbox, item["bbox"]):
                        if item["class_name"].startswith("no-") or item["class_name"] in ["fire", "smoke"]:
                            violations_this_frame.add(item["class_name"])

                # --- Temporal Filtering & Cooldown Logic ---
                if violations_this_frame:
                    state["violation_confirm_counter"] += 1
                    state["current_violations"].update(violations_this_frame)
                else:
                    # If no violation is seen, reset the counter
                    state["violation_confirm_counter"] = 0
                    state["current_violations"].clear()

                # Check if the violation is persistent enough to trigger an alert
                if state["violation_confirm_counter"] >= config.VIOLATION_CONFIRM_FRAMES:
                    current_time = time.time()
                    # Check if the cooldown period has passed
                    if (current_time - state["last_alert_time"]) > config.ALERT_COOLDOWN_SECONDS:
                        # Create and send the alert
                        alert_message = {
                            "timestamp": current_time,
                            "camera_id": camera_id,
                            "track_id": track_id,
                            "person_name": state["name"],
                            "violations": list(state["current_violations"]),
                            "bbox": person_bbox.tolist()
                        }
                        alert_queue.put(alert_message)

                        print(f"🚨 ALERT: {alert_message}")

                        # Update the state after sending the alert
                        state["last_alert_time"] = current_time
                        state["violation_confirm_counter"] = 0  # Reset counter to prevent spam

        except Exception as e:
            # In a real system, handle different queue empty/full exceptions
            # For now, we just print the error and continue
            # print(f"[Logic Engine] 🔴 An error occurred: {e}")
            pass

