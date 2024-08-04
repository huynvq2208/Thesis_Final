import numpy as np


embeddings = np.load('cke_embeddings.npz')

print(embeddings['user_embeddings'])