# import psycopg2
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from src.face_recognition.config import DB_CONFIG
#
# def connect_db():
#     return psycopg2.connect(**DB_CONFIG)
#
# def insert_embedding(name, embedding):
#     conn = connect_db()
#     cur = conn.cursor()
#     cur.execute("INSERT INTO face_embeddings (person_name, embedding) VALUES (%s, %s)", (name, list(embedding)))
#     conn.commit()
#     cur.close()
#     conn.close()
#
# def fetch_all_embeddings():
#     conn = connect_db()
#     cur = conn.cursor()
#     cur.execute("SELECT id, person_name, embedding FROM face_embeddings")
#     results = cur.fetchall()
#     cur.close()
#     conn.close()
#     return results
