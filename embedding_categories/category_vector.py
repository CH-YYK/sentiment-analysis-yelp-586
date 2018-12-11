import numpy as np

with open('embedding_categories/categories.embeddings', 'r') as f:
    category_vector = [[float(value) for value in line.split()] for line in f.readlines()][1:]
    category_vector.sort(key=lambda x: x[0])
    category_vector = np.array(category_vector)
    category_vector = np.concatenate([np.zeros([1, category_vector.shape[-1]]), category_vector], axis=0)
index, cat_vector = category_vector[:, 0], category_vector[:, 1:]
np.save("category_vector.npy", category_vector)