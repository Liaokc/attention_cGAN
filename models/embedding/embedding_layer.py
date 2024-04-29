import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer_EFA(nn.Module):
    """
    Graph_Attention_Layer_with_Edge_Features
    """
    def __init__(self, in_features, in_edge_features, out_features, dropout=0.10, alpha=0.20, lastact=False):
        """
        :param in_features: 节点特征维度
        :param in_edge_features: 边特征维度
        :param out_feature: 输出特征维度
        :param dropout: 权重暂退概率, 防止过拟合
        :param alpha: 超参数, 用于控制 LeakyReLU 激活函数负斜率
        :param lastact: 指示是否对最终输出应用激活函数
        """
        super(GraphAttentionLayer_EFA, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.lastact = lastact
        self.bn = torch.nn.BatchNorm1d(out_features)

        self.Wh = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.Wh1 = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.Wh2 = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.Wf = nn.Parameter(torch.zeros(size=(in_edge_features, out_features)))
        self.ah = nn.Parameter(torch.zeros(size=(out_features, 1)))
        self.af = nn.Parameter(torch.zeros(size=(out_features, 1)))
        self.bf = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_uniform_(self.Wh.data, gain=1.414)
        nn.init.xavier_uniform_(self.Wh1.data, gain=1.414)
        nn.init.xavier_uniform_(self.Wh2.data, gain=1.414)
        nn.init.xavier_uniform_(self.Wf.data, gain=1.414)
        nn.init.xavier_uniform_(self.ah.data, gain=1.414)
        nn.init.xavier_uniform_(self.af.data, gain=1.414)
        nn.init.xavier_uniform_(self.bf.data, gain=1.414)

        self.activation = nn.LeakyReLU(self.alpha)

    def forward(self, input, edge_feat, adj):
        """
        :param input: 节点特征矩阵 (N, num_of_nodes_features)
        :param edge_feat: 边特征矩阵
        :param adj: 邻接矩阵
        :return h_prime: 注意力分数矩阵 (N, N)
        """
        # 计算 h = input * W_h
        h = torch.mm(input, self.Wh)  # input: (N, in_features), W: (in_features, out_features), h: (N, out_features)
        N = h.size()[0]
        
        # 计算 cij
        h1 = torch.mm(input, self.Wh1)  # h1 (N, out_features)
        h2 = torch.mm(input, self.Wh2)  # h2 (N, out_features)

        # h1.repeat(1, N) (N, N * out_features)
        # h1.repeat(1, N).view(N * N, -1) (N * N, out_features)
        # ah_input (N * N, out_features)
        ah_input = h1.repeat(1, N).view(N * N, -1) + h2.repeat(N, 1)      # W_h*h_i + W_h*H_j
        ah_input = ah_input.view(N, -1, self.out_features)  # ah_input (N, N, out_features)
        # c (N, N)
        c = self.activation(torch.matmul(ah_input, self.ah).squeeze(2))  # (N, N, out_features) * (out_features, 1) = (N, N, 1) --> (N, N)
        
        # 计算 c'ij
        # (N, N, edge_features) --> (N, N, 1, edge_features)
        input_edge = edge_feat.unsqueeze(2)
        # f (N, N, 1, out_features)
        f = torch.matmul(input_edge, self.Wf)   #f  = W_f · f_ij
        # (N, N, 1, out_features)
        f = f + self.bf # f = W_f·f_ij + b_f
        # (N, N, 1, out_features)
        af_input = self.activation(f)   # af_input = δ(W_f·f_ij + b_f)
        cp = torch.matmul(af_input, self.af).squeeze(3) # cp = (N, N, 1, out_features) * (out_features, 1) = (N, N, 1, 1) --> (N, N, 1)
        # cp (N, N)
        cp = self.activation(cp.squeeze(2)) #cp = δ(a_f·δ(W_f·f_ij + b_f)))
        
        # 计算 cij & c'ij (N, N)
        c = c + cp
        
        # 计算 output = h * attention adj matrix
        zero_vec = -9e15*torch.ones_like(c)      # ones_like：返回大小与input相同的1张量
        attention = torch.where(adj>0, c, zero_vec)  
        attention = F.softmax(attention, dim=1)  #α_ij (N, N)
        #attention = F.dropout(attention, self.dropout, training=self.training)
        #原有dropout
        # h_prime (N, out_features)
        h_prime = torch.matmul(attention, h)     #=∑ α_ij · Wh · h_j 
        # （N, N）
        h_prime = self.bn(h_prime)
        if self.lastact == True:
            output = self.activation(h_prime)  # output = δ(∑ α_ij·Wh·h_j)
            return output
        else:
            output = h_prime  #=∑ α_ij·Wh·h_j
            return output
            
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
    

class OutputLayer(nn.Module):
    """
    GAT+EFA last layer
    将维度从in_features转变为out_features
    """
    def __init__(self, in_features, out_features):
        """
        :param in_features: 原始维度
        :param outfeatures: 嵌入层维度
        """
        super(OutputLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.activation = F.log_softmax
    
    def forward(self, input):
        output = torch.mm(input, self.W)  # (N, out_features)
        output = self.activation(output, dim=1)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
