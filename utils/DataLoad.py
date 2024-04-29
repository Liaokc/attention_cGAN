from gensim.models import KeyedVectors
import torch

def load_embedding(embedding_file_path):
    """加载嵌入向量文件"""
    embedding = KeyedVectors.load_word2vec_format(embedding_file_path)
    return embedding

def get_embedding_tensor(embedding_vecs, num_of_nodes, embedding_dim):
    embedding_tensor = torch.zeros(num_of_nodes, embedding_dim)

    for i in range(num_of_nodes):
        embedding_tensor[i] = torch.tensor(embedding_vecs[i])

    print("Embedding tensor shape:", embedding_tensor.shape)
    return embedding_tensor