from deepface import DeepFace

def get_embedding(face_img):
    return DeepFace.represent(face_img, model_name="ArcFace", detector_backend="skip")[0]["embedding"]
