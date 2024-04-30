import torch
import torch.nn as nn
import torch.nn.functional as F
from models.embedding.embedding_layer import GraphAttentionLayer_EFA


class GAT_EFA(nn.Module):
    """
    GAT with Edge Features 的高密度版本
    """
    def __init__(self, nfeat, nedgef, nhid, dropout, alpha, nheads, noutheads, nlayer):
        """
        Initialize GAT_EFA.

        :param nfeat: 节点特征维度
        :param nedgef: 边特征维度
        :param nclass: 最终输出维度
        :param nhid: 隐藏单元的数量, 每个注意力头输出的特征数量
        :param dropout: 权重暂退概率
        :param alpha: 超参数 LeakyReLU 负斜率
        :param nheads: 每个注意力层的注意力头的数量
        :param noutheads: 最后一层注意力层的输出头的数量
        :param nleayer: 模型的层数
        """
        super(GAT_EFA, self).__init__()
        self.dropout = dropout
        self.nlayer = nlayer
        self.nheads = nheads
        self.noutheads = noutheads
        
        # 初始化一个空列表，用于储存注意力层
        self.attentions = []
        # in layer
        # 用于储存注意力层
        """
        self.attentions 里面共有三个多头注意力层
        """
        # 增加第一个多头注意力层
        self.attentions.append([GraphAttentionLayer_EFA(nfeat, nedgef, nhid, dropout=dropout, alpha=alpha, lastact=True) for _ in range(nheads)])
        for i, attention in enumerate(self.attentions[0]):
            self.add_module(f"attention_{0}_{i}", attention)
        #attention layers
        # 增加第二个至 nlayer 个多头注意力层
        for j in range(1, nlayer-1):
            self.attentions.append([GraphAttentionLayer_EFA(nhid*nheads, nedgef, nhid, dropout=dropout, alpha=alpha, lastact=True) for _ in range(nheads)])
            for i, attention in enumerate(self.attentions[j]):
                self.add_module(f"attention_{j}_{i}", attention)
        #last attention layer
        # 增加最后一个多头注意力输出层
        self.attentions.append([GraphAttentionLayer_EFA(nhid*nheads, nedgef, nhid, dropout=dropout, alpha=alpha, lastact=False) for _ in range(noutheads)])
        for i, attention in enumerate(self.attentions[nlayer-1]):
            self.add_module(f"attention_{nlayer-1}_{i}", attention)
        # #output layer
        # self.out_layer = OutputLayer(nhid, nclass)
    
        # #self.activation = nn.LeakyReLU(alpha)
        # self.activation = F.relu

    def forward(self, x, edge_feats, adj):
        """
        Forward pass.
        
        :param x: 节点特征矩阵
        :param edge_feats: 边特征矩阵
        :param adj: 邻接矩阵
        :return: 模型的输出张量
        """
        # input_layer
        # output.shape = (N, nhid * nheads)
        # 每一层att的输出是 (N, nhid), 一共有nheads个, 最后cat在一起, 输出的shape是 (N, nhid * nheads)
        x = torch.cat([att(x, edge_feats, adj) for att in self.attentions[0]], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)  # output.shape = (N, nhid * nheads)
        # hidden layer
        for j in range(1, self.nlayer-1):
            _x = x
            # 输入 x 的 shape = (N, nhid * nheads), 每层att的输出 shape 是 (N, nhid)
            # 输出 x 的 shape 是 (N, nhid * nheads)
            x = torch.cat([att(x, edge_feats, adj) for att in self.attentions[j]], dim=1)
            x = x + _x  # residual connections残差连接 ((N, nhid * nheads)
            x = F.dropout(x, self.dropout, training=self.training)
        
        # last hidden layer
        # 输入的 x 的 shape = (N, nhid * nheads), 每层att的输出 shape 是 (N, nhid), stack后的 shape 是 (nheads, N, nhid)
        # mean 后的 shape = (N, nhid)
        x = torch.mean(torch.stack([att(x, edge_feats, adj) for att in self.attentions[self.nlayer-1]]), 0)
        # x = self.activation(x)  #h_i=δ(avg(∑ α_ij·Wh·h_j))
        # x = F.dropout(x, self.dropout, training=self.training)
        
        # # output layer
        # x = self.out_layer(x)
        return x