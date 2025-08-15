import time
from multiprocessing import Process, Queue
import config
from core.input_handler import capture_frames
from core.inference_engine import run_inference
from core.logic_engine import process_logic


def main():
    """
    Initializes and manages the multi-process surveillance pipeline.
    This is the main entry point of the application.
    """
    print("🚀 Starting Safety Surveillance System...")
    frame_queue = Queue(maxsize=config.FRAME_QUEUE_SIZE)
    results_queue = Queue()
    alert_queue = Queue()
    input_processes = []
    for camera_id, source_path in config.CAMERA_FEEDS.items():
        input_process = Process(
            target=capture_frames,
            args=(camera_id, source_path, frame_queue, config.TARGET_FPS),
            name=f"InputHandler-{camera_id}"
        )
        input_processes.append(input_process)
        input_process.start()
        print(f"   [Process Manager] Started Input Handler for Camera {camera_id}")
    inference_process = Process(
        target=run_inference,
        args=(frame_queue, results_queue),
        name="InferenceEngine"
    )
    inference_process.start()
    print(f"   [Process Manager] Started Inference Engine")
    logic_process = Process(
        target=process_logic,
        args=(results_queue, alert_queue),
        name="LogicEngine"
    )
    logic_process.start()
    print(f"   [Process Manager] Started Logic Engine")

    print("\n✅ All processes have been started. System is running.")
    print("   Press Ctrl+C in the terminal to stop the system.")

    try:
        while True:
            if not alert_queue.empty():
                alert = alert_queue.get()
                print(f"🚨 NEW ALERT RECEIVED: {alert}")
            all_processes = input_processes + [inference_process, logic_process]
            for p in all_processes:
                if not p.is_alive():
                    print(f"🔴 WARNING: Process {p.name} has terminated unexpectedly.")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Shutting down all processes...")
        all_processes = input_processes + [inference_process, logic_process]
        for p in all_processes:
            if p.is_alive():
                p.terminate()
                p.join()
        print("✅ System shut down gracefully.")


if __name__ == "__main__":
    main()