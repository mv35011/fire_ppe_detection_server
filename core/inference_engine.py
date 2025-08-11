import time
from multiprocessing import Queue
from ultralytics import YOLO
import torch
import config


def run_inference(frame_queue: Queue, results_queue: Queue):
    """
    A target function for the inference process, handling separate PPE and Fire models.
    """
    print("[Inference Engine] 🟢 Starting...")

    # Load both models
    print(f"[Inference Engine] Loading PPE model: {config.PPE_MODEL_PATH}")
    ppe_model = YOLO(config.PPE_MODEL_PATH)

    print(f"[Inference Engine] Loading Fire model: {config.FIRE_MODEL_PATH}")
    fire_model = YOLO(config.FIRE_MODEL_PATH)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ppe_model.to(device)
    fire_model.to(device)
    print(f"[Inference Engine] ✅ Models loaded successfully on device: {device.upper()}.")

    while True:
        # 1. Batch Collection (No changes here)
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
            continue

        # 2. Sequential Batched Inference
        try:
            # Run the first model
            ppe_results = ppe_model.predict(source=frames_batch, conf=config.CONF_THRESHOLD, verbose=False)
            # Run the second model
            fire_results = fire_model.predict(source=frames_batch, conf=config.CONF_THRESHOLD, verbose=False)
        except Exception as e:
            print(f"[Inference Engine] 🔴 ERROR during model prediction: {e}")
            continue

        # 3. Combine and Queue Results
        for i, (ppe_res, fire_res) in enumerate(zip(ppe_results, fire_results)):
            camera_id = camera_ids_batch[i]

            # Combine all detections into a single list
            all_detections = []

            # Add PPE detections (assuming class names are in the model)
            for box in ppe_res.boxes:
                all_detections.append({
                    "bbox": box.xyxy[0].cpu().numpy(),
                    "score": box.conf[0].cpu().numpy(),
                    "class_id": int(box.cls[0].cpu().numpy()),
                    "class_name": ppe_model.names[int(box.cls[0])]
                })

            # Add Fire/Smoke detections
            for box in fire_res.boxes:
                all_detections.append({
                    "bbox": box.xyxy[0].cpu().numpy(),
                    "score": box.conf[0].cpu().numpy(),
                    "class_id": int(box.cls[0].cpu().numpy()),
                    "class_name": fire_model.names[int(box.cls[0])]
                })

            # This is the unified data structure we send to the logic engine
            output_data = {
                "camera_id": camera_id,
                "original_frame": frames_batch[i],
                "detections": all_detections  # A single list with all objects
            }

            results_queue.put(output_data)
