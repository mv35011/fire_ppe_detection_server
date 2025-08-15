import time
from multiprocessing import Queue
from ultralytics import YOLO
import torch
import config


def run_inference(frame_queue: Queue, results_queue: Queue):
    """
    A target function for the inference process, handling three separate models.
    This is a temporary prototype setup. The ideal solution is a single unified model.
    """
    print("[Inference Engine] 🟢 Starting...")
    print(f"[Inference Engine] Loading Person model: {config.PERSON_MODEL_PATH}")
    person_model = YOLO(config.PERSON_MODEL_PATH)

    print(f"[Inference Engine] Loading PPE model: {config.PPE_MODEL_PATH}")
    ppe_model = YOLO(config.PPE_MODEL_PATH)

    print(f"[Inference Engine] Loading Fire model: {config.FIRE_MODEL_PATH}")
    fire_model = YOLO(config.FIRE_MODEL_PATH)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    person_model.to(device)
    ppe_model.to(device)
    fire_model.to(device)
    print(f"[Inference Engine] ✅ All models loaded successfully on device: {device.upper()}.")

    while True:
        frames_batch = []
        camera_ids_batch = []

        while len(frames_batch) < config.INFERENCE_BATCH_SIZE:
            try:
                camera_id, frame = frame_queue.get(timeout=0.01)
                frames_batch.append(frame)
                camera_ids_batch.append(camera_id)
            except Exception:
                break

        if not frames_batch:
            time.sleep(0.01)
            continue
        print(f"[Inference Engine] [DEBUG] Collected a batch of {len(frames_batch)} frames.")
        try:
            person_results = person_model.predict(source=frames_batch, classes=[0], conf=config.CONF_THRESHOLD,
                                                  verbose=False)
            ppe_results = ppe_model.predict(source=frames_batch, conf=config.CONF_THRESHOLD, verbose=False)
            fire_results = fire_model.predict(source=frames_batch, conf=config.CONF_THRESHOLD, verbose=False)
            print(f"[Inference Engine] [DEBUG] Finished prediction on batch.")

        except Exception as e:
            print(f"[Inference Engine] 🔴 ERROR during model prediction: {e}")
            continue
        for i in range(len(frames_batch)):
            camera_id = camera_ids_batch[i]

            all_detections = []
            for box in person_results[i].boxes:
                all_detections.append({
                    "bbox": box.xyxy[0].cpu().numpy(), "score": box.conf[0].cpu().numpy(),
                    "class_name": person_model.names[int(box.cls[0])]
                })
            for box in ppe_results[i].boxes:
                all_detections.append({
                    "bbox": box.xyxy[0].cpu().numpy(), "score": box.conf[0].cpu().numpy(),
                    "class_name": ppe_model.names[int(box.cls[0])]
                })
            for box in fire_results[i].boxes:
                all_detections.append({
                    "bbox": box.xyxy[0].cpu().numpy(), "score": box.conf[0].cpu().numpy(),
                    "class_name": fire_model.names[int(box.cls[0])]
                })

            output_data = {
                "camera_id": camera_id,
                "original_frame": frames_batch[i],
                "detections": all_detections
            }
            results_queue.put(output_data)
            if all_detections:
                print(f"[Inference Engine] [DEBUG] Queued {len(all_detections)} detections for camera {camera_id}.")
