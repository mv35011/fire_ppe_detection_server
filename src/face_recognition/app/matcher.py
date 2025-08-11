import numpy as np
import sys 
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.face_recognition.config import THRESHOLD

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_match(embedding, db_embeddings):
    for db_id, db_name, db_emb in db_embeddings:
        sim = cosine_similarity(embedding, db_emb)
        if sim > (1 - THRESHOLD):
            return True, db_name
    return False, None
