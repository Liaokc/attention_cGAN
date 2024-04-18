import numpy as np
import torch
# data_matrix是6674*98丰度信息矩阵
# num_samples是98个样本的总数

# 设置划分比例
def split_data_random(data_matrix: torch.tensor):
    train_ratio = 0.7
    val_ratio = 0.15
    # test_ratio = 0.15
    num_samples = data_matrix.shape[1]

    # 计算每个集合的大小
    num_train = int(num_samples * train_ratio)
    num_val = int(num_samples * val_ratio)
    num_test = num_samples - num_train - num_val

    # 随机打乱样本索引
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    # 划分数据集
    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_val]
    test_indices = indices[num_train + num_val:]

    # 使用划分的索引来选择对应的样本
    train_data = data_matrix[train_indices, :]
    val_data = data_matrix[val_indices, :]
    test_data = data_matrix[test_indices, :]
    return train_data, val_data, test_data

if __name__=="__main__":
    import pickle
    with open('data/raw/combo.pkl', 'rb') as f:
        my_data = pickle.load(f)

    otu_tab = my_data["otu_tab(df)"]
    df = otu_tab.sort_index(axis=1)
    df = df.sort_index()
    otu_tab_tensor_data = torch.tensor(df.values)
    train_data, val_data, test_data = split_data_random(otu_tab_tensor_data)