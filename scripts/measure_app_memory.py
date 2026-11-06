import os
import psutil
import time

p = psutil.Process(os.getpid())
print(f"Initial RSS: {p.memory_info().rss / (1024*1024):.2f} MB", flush=True)

import torch
print(f"After import torch: {p.memory_info().rss / (1024*1024):.2f} MB", flush=True)

from sentence_transformers import SentenceTransformer
print(f"After import sentence_transformers: {p.memory_info().rss / (1024*1024):.2f} MB", flush=True)

t0 = time.time()
sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print(f"After SentenceTransformer load ({time.time()-t0:.2f}s): {p.memory_info().rss / (1024*1024):.2f} MB", flush=True)

import joblib
xgb_model = joblib.load("ats_xgboost_model.pkl")
print(f"After xgb_model load: {p.memory_info().rss / (1024*1024):.2f} MB", flush=True)

import numpy as np
resume = "Experienced Senior Software Engineer with 8 years in Python, machine learning, distributed systems, and Kubernetes."
jd = "Looking for a Senior Python Developer with expertise in ML pipelines, Docker, AWS, and strong communication skills."

r_emb = sbert_model.encode([resume])
j_emb = sbert_model.encode([jd])
feat = np.concatenate([r_emb, j_emb], axis=1)
score = xgb_model.predict(feat)[0]
print(f"Prediction result: {float(score):.2f}", flush=True)
print(f"Peak RSS after prediction: {p.memory_info().rss / (1024*1024):.2f} MB", flush=True)

