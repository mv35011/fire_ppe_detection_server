from deepface import DeepFace

backend = "retinaface"

def detect_faces(frame):
    detections = DeepFace.extract_faces(
        img_path=frame,
        detector_backend=backend,
        enforce_detection=False,
        align=True
    )

    result = []
    for face in detections:
        face_img = face["face"]
        x = face["facial_area"]["x"]
        y = face["facial_area"]["y"]
        w = face["facial_area"]["w"]
        h = face["facial_area"]["h"]
        result.append((face_img, (x, y, w, h)))

    return result
