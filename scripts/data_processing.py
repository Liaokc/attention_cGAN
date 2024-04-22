import torch

class TreeMatrixMessage:
    def __init__(self, tree_data):
        # super(TreeMatrixMessage, self).__init__()
        self.tree_data = tree_data
        self.nodes_number = self.get_all_node_nums()

    def create_indexed_tensor(self):
        """
        创建一个索引张量，用于填充邻接矩阵或边长矩阵。
        :param tree_data: 包含系统发育树数据的字典。
        :return: 一个形状为 [边的数量, 2] 的张量，其中包含边的索引。
        """
        Edge = self.tree_data["Edge"]
        indices = torch.tensor(Edge)
        return indices

    def make_adjacency_matrix(self):
        """创建邻接矩阵"""
        indices = self.create_indexed_tensor()
        adjacency_matrix = torch.zeros(self.nodes_number, self.nodes_number, dtype=torch.float)

        adjacency_matrix[indices[:, 0] - 1, indices[:, 1] - 1] = 1
        return adjacency_matrix

    def make_edge_lengths_matrix(self, normalization=None):
        """创建边长矩阵"""
        Edge_Lengths = torch.tensor(self.tree_data["Edge_Lengths"])
        indices = self.create_indexed_tensor()
        
        # 边长归一化
        if normalization:
            Edge_Lengths = (Edge_Lengths - Edge_Lengths.mean()) / Edge_Lengths.std()

        # 需要确保 Edge_Lengths 的数据类型与 adjacency_matrix 一致
        edge_lengths_matrix = torch.zeros(self.nodes_number, self.nodes_number, dtype=torch.float)
        # _Edge_Lengths = Edge_Lengths.clone().detach()
        edge_lengths_matrix[indices[:, 0] - 1, indices[:, 1] - 1] = torch.tensor(Edge_Lengths.clone().detach())
        return edge_lengths_matrix

    def make_tree_matrix(self):
        """创建树矩阵, 包含邻接矩阵和边长矩阵"""
        adjacency_matrix = self.make_adjacency_matrix()
        edge_lengths_matrix = self.make_edge_lengths_matrix()
        return torch.where(adjacency_matrix == 1, edge_lengths_matrix, float('inf'))
    

    def get_all_node_nums(self):
        """获得节点数量"""
        edge = self.tree_data["Edge"]
        
        node_set = {node for every_edge in edge for node in every_edge}
        return len(node_set)

    def get_edges_count(self):
        """获得边的数量"""
        edge_lengths = self.tree_data["Edge_Lengths"]
        
        return len(edge_lengths)