import pandas as pd
from collections import defaultdict
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt
import pickle

categories = pd.read_csv('../data/business_reviews2017.tsv', sep='\t', usecols=['categories'])['categories']
cates = [cate.split(', ') for cate in categories]

# corpus of categories
corpus = set()
for i in cates:
    corpus = corpus.union(i)
corpus = {j: str(i+1) for i, j in enumerate(corpus)}

# save corpus
pickle.dump(corpus, open('category_corpus.pkl', 'wb'))

# tokenize tags
categories = [[corpus.get(tag) for tag in cate] for cate in cates]

def get_cat_edge_list(categories):
    """
    :param categories: list of list that contain tuples which inger edges
    :return: edge-list for network
    """
    edges = [list(combinations(comb, 2)) for comb in categories]
    edge_list = []
    # edge list split by '|'
    for i in edges:
        edge_list += [' '.join(edge) for edge in i]
    return edge_list

# get edge list
edge_list = get_cat_edge_list(categories)

# networkx: build graph by parsing edge list
G = nx.parse_edgelist(edge_list, delimiter=' ')

# networkx: create figure
nx.draw_networkx(G, with_labels=False, alpha=0.5)
limits = plt.axis('off')

# networks: output adjacent matrix
with open('categories.adjlist', 'w') as f:
    f.writelines([i+'\n' for i in edge_list])
