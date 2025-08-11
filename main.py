import time
from multiprocessing import Process, Queue
import config

# Import the target functions from the core modules
from core.input_handler import capture_frames
from core.inference_engine import run_inference
from core.logic_engine import process_logic


def main():
    """
    Initializes and manages the multi-process surveillance pipeline.
    This is the main entry point of the application.
    """
    print("🚀 Starting Safety Surveillance System...")

    # 1. Create the shared queues for communication between processes
    #    - frame_queue: Input handlers send raw frames here.
    #    - results_queue: Inference engine sends detection results here.
    #    - alert_queue: Logic engine sends final alerts here for the dashboard.
    frame_queue = Queue(maxsize=config.FRAME_QUEUE_SIZE)
    results_queue = Queue()
    alert_queue = Queue()

    # 2. Start the Input Handler processes
    #    One process is started for each camera feed defined in the config.
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

    # 3. Start the Inference Engine process
    inference_process = Process(
        target=run_inference,
        args=(frame_queue, results_queue),
        name="InferenceEngine"
    )
    inference_process.start()
    print(f"   [Process Manager] Started Inference Engine")

    # 4. Start the Logic Engine process
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
        # Keep the main script alive to monitor child processes
        # and handle alerts from the final queue.
        while True:
            # You can add logic here to consume alerts from the alert_queue
            # and send them to a dashboard or database.
            if not alert_queue.empty():
                alert = alert_queue.get()
                print(f"🚨 NEW ALERT RECEIVED: {alert}")

            # Optional: Monitor the health of child processes
            all_processes = input_processes + [inference_process, logic_process]
            for p in all_processes:
                if not p.is_alive():
                    print(f"🔴 WARNING: Process {p.name} has terminated unexpectedly.")
                    # In a production system, you would add logic here to restart it.

            time.sleep(1)  # Check for alerts once per second

    except KeyboardInterrupt:
        print("\n🛑 Shutting down all processes...")
        # Terminate all child processes gracefully
        all_processes = input_processes + [inference_process, logic_process]
        for p in all_processes:
            if p.is_alive():
                p.terminate()
                p.join()
        print("✅ System shut down gracefully.")


if __name__ == "__main__":
    # The 'if __name__ == "__main__"' guard is crucial for multiprocessing
    # to prevent child processes from re-importing and re-running the main script.
    main()
