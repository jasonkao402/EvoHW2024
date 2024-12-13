import numpy as np
# for _ in range(100):
#     print(np.random.geometric(0.333))
# cosine similarity
a = np.array([4, 3])
b = np.array([1, 0])
print(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))