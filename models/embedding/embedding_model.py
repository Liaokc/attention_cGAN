import torch
import torch.nn as nn
import torch.nn.functional as F
from models.embedding.embedding_layer import GraphAttentionLayer_EFA


class GAT_EFA(nn.Module):
    def __init__(self, nfeat, nedgef, nhid, nclass, dropout, alpha, nheads, noutheads, nlayer):
        """
        Dense version of GAT.
        :param nfeat: 节点特征维度
        :param nedgef: 边特征维度
        :param nclass: 最终输出维度
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
            self.add_module('attention_{}_{}'.format(0, i), attention)
        #attention layers
        # 增加第二个多头注意里层
        for j in range(1, nlayer-1):
            self.attentions.append([GraphAttentionLayer_EFA(nhid*nheads, nedgef, nhid, dropout=dropout, alpha=alpha, lastact=True) for _ in range(nheads)])
            for i, attention in enumerate(self.attentions[j]):
                self.add_module('attention_{}_{}'.format(j, i), attention)
        #last attention layer
        # 增加最后一个多头注意力层
        self.attentions.append([GraphAttentionLayer_EFA(nhid*nheads, nedgef, nhid, dropout=dropout, alpha=alpha, lastact=False) for _ in range(noutheads)])
        for i, attention in enumerate(self.attentions[nlayer-1]):
            self.add_module('attention_{}_{}'.format(nlayer-1, i), attention)
        # #output layer
        # self.out_layer = OutputLayer(nhid, nclass)
    
        # #self.activation = nn.LeakyReLU(alpha)
        # self.activation = F.relu

    def forward(self, x, edge_feats, adj):
        # input_layer
        x = torch.cat([att(x, edge_feats, adj) for att in self.attentions[0]], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        # hidden layer
        for j in range(1, self.nlayer-1):
            mid = torch.cat([att(x, edge_feats, adj) for att in self.attentions[j]], dim=1)
            x = mid + x  #residual connections
            x = F.dropout(x, self.dropout, training=self.training)
        
        # last hidden layer
        x = torch.mean(torch.stack([att(x, edge_feats, adj) for att in self.attentions[self.nlayer-1]]), 0)
        # x = self.activation(x)  #h_i=δ(avg(∑ α_ij·Wh·h_j))
        # x = F.dropout(x, self.dropout, training=self.training)
        
        # # output layer
        # x = self.out_layer(x)
        return x