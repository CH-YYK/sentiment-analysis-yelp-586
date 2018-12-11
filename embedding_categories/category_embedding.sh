# need export...
edgelist=./embedding_categories/categories.edgelist
embedding=./embedding_categories/categories.embeddings

#
deepwalk --format edgelist --input "${edgelist}" --max-memory-data-size 0 --number-walks 80 --representation-size 200 --walk-length 40 --window-size 10 --workers 2 --output $embedding