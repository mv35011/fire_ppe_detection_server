import time
from multiprocessing import Process, Queue
import config  # We will create this file for settings

# Import the function from the core module
from core.input_handler import capture_frames
# We will create these modules in the next steps
# from core.inference_engine import run_inference
# from core.logic_engine import process_logic

def main():
    """
    Initializes and manages the multi-process surveillance pipeline.
    """
    print("🚀 Starting Safety Surveillance System...")

    # 1. Create the shared queues for communication between processes
    #    - frame_queue: Input handlers send raw frames here.
    #    - results_queue: Inference engine sends detection results here.
    #    - alert_queue: Logic engine sends final alerts here.
    frame_queue = Queue(maxsize=config.FRAME_QUEUE_SIZE)
    results_queue = Queue()
    alert_queue = Queue()

    # 2. Start the Input Handler processes
    #    One process is started for each camera feed defined in the config.
    input_processes = []
    for camera_id, source_path in config.CAMERA_FEEDS.items():
        input_process = Process(
            target=capture_frames,
            args=(camera_id, source_path, frame_queue, config.TARGET_FPS)
        )
        input_processes.append(input_process)
        input_process.start()
        print(f"   [Process Manager] Started Input Handler for Camera {camera_id}")

    # 3. Start the Inference Engine process (To be implemented)
    # inference_process = Process(
    #     target=run_inference,
    #     args=(frame_queue, results_queue, config.MODEL_PATH)
    # )
    # inference_process.start()
    # print(f"   [Process Manager] Started Inference Engine")

    # 4. Start the Logic Engine process (To be implemented)
    # logic_process = Process(
    #     target=process_logic,
    #     args=(results_queue, alert_queue)
    # )
    # logic_process.start()
    # print(f"   [Process Manager] Started Logic Engine")

    print("\n✅ All processes have been started. System is running.")
    print("   Press Ctrl+C to stop the system.")

    try:
        # Keep the main script alive to manage the child processes
        while True:
            # Here you could add logic to monitor the queues or process alerts
            # For now, we just check if any process has died unexpectedly.
            for p in input_processes:
                if not p.is_alive():
                    print(f"🔴 WARNING: Process {p.name} has terminated unexpectedly.")
                    # Here you could add logic to restart the process
            time.sleep(10)

    except KeyboardInterrupt:
        print("\n🛑 Shutting down all processes...")
        for p in input_processes:
            p.terminate()
            p.join()
        # inference_process.terminate()
        # inference_process.join()
        # logic_process.terminate()
        # logic_process.join()
        print("✅ System shut down gracefully.")


if __name__ == "__main__":
    # This ensures that child processes don't re-run the main script on import
    main()
