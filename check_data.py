import numpy as np
import os

# --- Configuration ---
# Make sure these paths are correct
EMBEDDINGS_PATH = "data/embeddings.npy"
NAMES_PATH = "data/n.npy"

# --- Script ---
print(f"--- Inspecting Face Recognition Data ---")

if os.path.exists(EMBEDDINGS_PATH):
    embeddings = np.load(EMBEDDINGS_PATH)
    print(f"\n✅ Found embeddings file: {EMBEDDINGS_PATH}")
    print(f"   - Shape: {embeddings.shape}")
    print(f"   - Number of embeddings: {len(embeddings)}")
else:
    print(f"\n🔴 ERROR: Embeddings file not found at '{EMBEDDINGS_PATH}'")

if os.path.exists(NAMES_PATH):
    names = np.load(NAMES_PATH, allow_pickle=True)
    print(f"\n✅ Found names file: {NAMES_PATH}")
    print(f"   - Number of names: {len(names)}")
    print(f"   - Contents: {names}") # This will print all the names
else:
    print(f"\n🔴 ERROR: Names file not found at '{NAMES_PATH}'")