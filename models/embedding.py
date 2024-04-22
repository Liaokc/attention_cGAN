from node2vec import Node2Vec
from torch import nn

def get_EmbeddingVec(graph, embedding_dim, walk_length, num_walks, window, min_count, batch_words, p=1, q=1, weight_key=None, workers=1):

    node2vec = Node2Vec(graph, dimensions=embedding_dim, walk_length=walk_length, num_walks=num_walks, p=p, q=q, weight_key=weight_key, workers=workers)

    # 训练模型
    model = node2vec.fit(window=window, min_count=min_count, batch_words=batch_words)

    # 获取嵌入向量
    embedding = {node: model.wv[node] for node in graph.nodes()}

    return embedding

if __name__=="__main__":
    import pickle
    with open("data/raw/TreeRooted_graph.pkl", "rb") as f:
        graph = pickle.load(f)

    embedding = get_EmbeddingVec(graph, 128, 80, 300, 0.5, 2, "weight", 4, 10, 1, 4)
    print(embedding)