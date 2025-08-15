import os
import shutil
import time
import json
from typing import List
from multiprocessing import Process, Queue
from fastapi import FastAPI, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
import uuid
from core.input_handler import capture_frames
from core.inference_engine import run_inference
from core.logic_engine import process_logic
import config

app = FastAPI(title="Video Surveillance API")
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_pipeline(video_paths: List[str], request_id: str):
    """
    This function encapsulates the entire surveillance pipeline.
    It is run as a background process.
    """
    print(f"🚀 Starting pipeline for request ID: {request_id} with videos: {video_paths}")

    frame_queue = Queue(maxsize=config.FRAME_QUEUE_SIZE)
    results_queue = Queue()
    alert_queue = Queue()
    camera_feeds = {i: path for i, path in enumerate(video_paths)}
    input_processes = [
        Process(
            target=capture_frames,
            args=(camera_id, source_path, frame_queue, config.TARGET_FPS),
            name=f"InputHandler-{camera_id}"
        )
        for camera_id, source_path in camera_feeds.items()
    ]
    for p in input_processes:
        p.start()
        print(f"    [Process Manager] Started Input Handler for Camera {p.name}")
    inference_process = Process(
        target=run_inference,
        args=(frame_queue, results_queue),
        name="InferenceEngine"
    )
    inference_process.start()
    print("    [Process Manager] Started Inference Engine")
    logic_process = Process(
        target=process_logic,
        args=(results_queue, alert_queue),
        name="LogicEngine"
    )
    logic_process.start()
    print("    [Process Manager] Started Logic Engine")
    all_processes = input_processes + [inference_process, logic_process]
    results_file_path = os.path.join(RESULTS_DIR, f"{request_id}.json")

    try:
        while any(p.is_alive() for p in all_processes):
            while not alert_queue.empty():
                try:
                    alert = alert_queue.get(timeout=1)
                    with open(results_file_path, 'a') as f:
                        f.write(json.dumps(alert) + '\n')
                    print(f"📦 Wrote new alert to file for request {request_id}")
                except Exception as e:
                    print(f"🔴 Error writing alert to file: {e}")
            time.sleep(1)

    except Exception as e:
        print(f"🔴 WARNING: Pipeline for request {request_id} encountered an error: {e}")
    finally:
        print(f"🛑 Shutting down all processes for request ID: {request_id}...")
        for p in all_processes:
            if p.is_alive():
                p.terminate()
                p.join()
        for path in video_paths:
            if os.path.exists(path):
                os.remove(path)
        shutil.rmtree(os.path.join("temp_videos", request_id), ignore_errors=True)

        print("✅ System shut down gracefully and temporary files cleaned up.")


@app.post("/analyze_videos/")
async def analyze_videos(
        background_tasks: BackgroundTasks,
        files: List[UploadFile]
):
    """
    Receives one or more video files and starts the surveillance pipeline.
    The videos are saved to a temporary directory and the processing is
    handled in a background task.
    """
    if not files:
        return JSONResponse(status_code=400, content={"message": "No files uploaded."})
    request_id = str(uuid.uuid4())
    temp_dir = os.path.join("temp_videos", request_id)
    os.makedirs(temp_dir, exist_ok=True)
    video_paths = []

    try:
        for file in files:
            file_location = os.path.join(temp_dir, file.filename)
            video_paths.append(file_location)
            with open(file_location, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        background_tasks.add_task(run_pipeline, video_paths, request_id)
        with open(os.path.join(RESULTS_DIR, f"{request_id}.json"), 'w') as f:
            pass

        return JSONResponse(
            status_code=202,
            content={
                "message": "Video analysis started successfully in the background. Check /results/{request_id} for alerts.",
                "request_id": request_id,
                "video_names": [file.filename for file in files]
            }
        )

    except Exception as e:
        shutil.rmtree(temp_dir)
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {e}"})


@app.get("/results/{request_id}")
async def get_results(request_id: str):
    """
    Endpoint to retrieve alerts for a specific request ID.
    This will block until an alert is available or a timeout occurs.
    """
    results_file_path = os.path.join(RESULTS_DIR, f"{request_id}.json")
    if not os.path.exists(results_file_path):
        return JSONResponse(status_code=404, content={
            "message": "Invalid or expired request ID. The analysis may not have started yet or has finished."})

    try:
        alerts = []
        with open(results_file_path, 'r') as f:
            for line in f:
                alerts.append(json.loads(line))
        if alerts:
            return JSONResponse(content={"request_id": request_id, "alerts": alerts})
        else:
            return JSONResponse(content={"message": "Processing in progress. No new alerts yet."})

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"An error occurred while fetching results: {e}"})


if __name__ == "__main__":
    import uvicorn
    os.makedirs("temp_videos", exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)