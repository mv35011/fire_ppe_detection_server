import cv2
import time
from multiprocessing import Queue


def capture_frames(camera_id: int, source_path: str, frame_queue: Queue, target_fps: int):
    """
    A target function for a process that continuously reads frames from a video source.

    This function is designed to run in its own process for each camera feed. It reads
    frames, resizes them for consistency, and puts them into a shared queue for the
    inference engine to process.

    Args:
        camera_id (int): A unique identifier for this camera feed (e.g., 0, 1, 2).
        source_path (str): The path to the video file or the camera stream URL.
        frame_queue (Queue): The shared multiprocessing queue to send frames to.
        target_fps (int): The desired frames per second to process from the video.
    """
    print(f"[Input Handler {camera_id}] 🟢 Starting...")

    # Calculate the delay needed to match the target FPS
    frame_delay = 1 / target_fps

    while True:
        # Open the video capture object. This is inside the loop to allow
        # automatic reconnection if a stream drops or a video ends.
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            print(
                f"[Input Handler {camera_id}] 🔴 ERROR: Could not open video source: {source_path}. Retrying in 5 seconds...")
            time.sleep(5)
            continue

        print(f"[Input Handler {camera_id}] ✅ Video source opened successfully.")

        while True:
            start_time = time.time()

            ret, frame = cap.read()

            # If the video ends or the frame is not read correctly,
            # break the inner loop to reopen the video source (for looping).
            if not ret:
                print(f"[Input Handler {camera_id}] 🔄 Video ended. Re-opening...")
                break

            try:
                # Put the frame and its source camera ID into the queue.
                # The 'block=False' will raise an exception if the queue is full,
                # which helps prevent memory overload if the inference engine is slow.
                frame_queue.put((camera_id, frame), block=False)
            except Exception as e:
                # This can happen if the queue is full. We can just skip the frame.
                # print(f"[Input Handler {camera_id}] 🟡 WARNING: Frame queue is full. Skipping frame.")
                pass

            # Control the frame rate by waiting for the calculated delay
            elapsed_time = time.time() - start_time
            sleep_time = frame_delay - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Release the capture object before reopening
        cap.release()

