from gensim.models import KeyedVectors

def load_embedding(embedding_file_path):
    """加载嵌入向量文件"""
    embedding = KeyedVectors.load_word2vec_format(embedding_file_path)
    return embedding