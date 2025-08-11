# Real-Time Multi-Feed Safety Surveillance System

Important Weights can be  downloaded here(yolov9)- https://drive.google.com/drive/folders/1RNeJYNjcLuC2LYyT2Hj-qLICvUzxY_cR?usp=sharing




A high-performance, AI-powered surveillance platform designed to enhance workplace safety through real-time monitoring and automated threat detection.

## Overview

This system monitors multiple video feeds simultaneously to automatically detect:
- **PPE compliance violations**
- **Fire and smoke hazards**
- **Unsafe or suspicious behavior** (via person tracking and facial recognition)

The architecture is optimized for **low latency** and **high throughput**, using a multi-process producer-consumer pipeline to efficiently handle concurrent camera streams.

## Key Features

- **Multi-Feed Processing** – Simultaneous, non-blocking analysis of multiple video streams
- **Unified Object Detection** – Single YOLOv9 model detects all relevant classes (person, PPE, fire, smoke) in one pass
- **Person Tracking** – ByteTrack assigns consistent IDs to individuals across frames
- **Face Recognition** – Matches tracked individuals against a database of known embeddings for identification
- **Smart Violation Logic** – Temporal filtering & cooldowns to reduce false positives
- **Dashboard (Planned)** – Web-based interface for real-time alerts and system status

## Architecture

The system uses a **decoupled, multi-process design** to prevent bottlenecks:

```
[Cam 1] → Input Process 1 ┐
[Cam 2] → Input Process 2 ├──> Shared Frame Queue → Inference Engine → Results Queue → Logic Engine → Alerts
[Cam 3] → Input Process 3 ┘
```

### Process Roles

- **`main.py`** – Orchestrates and monitors all worker processes
- **`core/input_handler.py`** – One process per camera; captures frames → `frame_queue`
- **`core/inference_engine.py`** – Runs batched GPU inference → `results_queue`
- **`core/logic_engine.py`** – Applies tracking, face recognition, violation checks, and alert cooldowns

## Quick Start

### Prerequisites
- Python 3.8+
- GPU with CUDA support (recommended)
- Required Python packages

### Installation

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure settings**
   Update `config.py` with:
   - Camera feed URLs/paths
   - Model file paths
   - Face recognition database path

3. **Start the system**
   ```bash
   python main.py
   ```

## Configuration

Edit `config.py` to customize:
- Camera sources (RTSP streams, local files, or webcams)
- Detection thresholds
- Alert cooldown periods
- Face recognition database location
- Output settings

## Project Structure

```
├── main.py                 # Main orchestrator
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── core/
│   ├── input_handler.py   # Camera feed processing
│   ├── inference_engine.py# AI model inference
│   └── logic_engine.py    # Business logic & alerts
├── models/                # AI model files
└── database/             # Face recognition database
```

## Roadmap

### Planned Enhancements
- **Unified Model Training** – Merge PPE, fire/smoke, and person detection into one optimized model
- **Database Logging** – Store violation events in PostgreSQL/SQLite for history and analytics
- **Live Dashboard** – Streamlit/Flask app for monitoring alerts and feed status
- **Async Face Recognition** – Offload face recognition to a separate worker pool
- **Dynamic Camera Management** – Add/remove feeds without restarting the system
- **Alert Notifications** – Email/SMS alerts for high-priority events (e.g., fire detection)

## Performance

The system is designed to handle:
- Multiple concurrent video streams (tested up to 8 feeds)
- Real-time processing with minimal latency
- Efficient GPU utilization through batched inference
- Smart resource management to prevent memory leaks

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

 issues and questions, please [create an issue](link-to-issues) or contact the development team.