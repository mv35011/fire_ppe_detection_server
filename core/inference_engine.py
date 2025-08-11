import time
from multiprocessing import Queue
from ultralytics import YOLO
import torch
import config


def run_inference(frame_queue: Queue, results_queue: Queue):
    """
    A target function for the inference process.

    This function continuously pulls frames from the input queue, performs
    batched inference using a YOLO model, and puts the results into an
    output queue for the logic engine.

    Args:
        frame_queue (Queue): The shared queue from which to pull raw frames.
        results_queue (Queue): The shared queue to push detection results to.
    """
    print("[Inference Engine] 🟢 Starting...")

    # Load the YOLO model inside the process
    # This is crucial as model objects cannot be shared across processes.
    print(f"[Inference Engine] Loading model: {config.MODEL_PATH}")
    model = YOLO(config.MODEL_PATH)
    print("[Inference Engine] ✅ Model loaded successfully.")

    # Set device to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"[Inference Engine] Using device: {device.upper()}")

    while True:
        # 1. Batch Collection
        frames_batch = []
        camera_ids_batch = []
        original_frames_batch = []

        # Pull frames from the queue until the batch is full or the queue is empty
        while len(frames_batch) < config.INFERENCE_BATCH_SIZE:
            try:
                # Get a frame without waiting indefinitely
                camera_id, frame = frame_queue.get(timeout=0.01)
                frames_batch.append(frame)
                camera_ids_batch.append(camera_id)
            except Exception:
                # If the queue is empty, break the loop to process what we have
                break

        # If no frames were collected, just continue to the next iteration
        if not frames_batch:
            continue

        # 2. Batched Inference
        try:
            results = model.predict(
                source=frames_batch,
                conf=config.CONF_THRESHOLD,
                iou=config.IOU_THRESHOLD,
                verbose=False
            )
        except Exception as e:
            print(f"[Inference Engine] 🔴 ERROR during model prediction: {e}")
            continue

        # 3. Process and Queue Results
        for i, result in enumerate(results):
            camera_id = camera_ids_batch[i]

            # This is the data structure we will send to the logic engine
            output_data = {
                "camera_id": camera_id,
                "original_frame": frames_batch[i],  # Send the original frame for cropping
                "detections": []
            }

            # Extract bounding boxes, scores, and class IDs
            for box in result.boxes:
                output_data["detections"].append({
                    "bbox": box.xyxy[0].cpu().numpy(),
                    "score": box.conf[0].cpu().numpy(),
                    "class_id": int(box.cls[0].cpu().numpy())
                })

            # Put the processed data into the results queue
            results_queue.put(output_data)

