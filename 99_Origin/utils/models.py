import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_mean_pool


# Thomas N. Kipf and Max Welling,
# Semi-Supervised Classification with Graph Convolutional Networks
# International Conference on Learning Representations (ICLR) 2017
class GCN(nn.Module):
    # graph conovolutional network (GCN)
    def __init__(self, num_node_feats, dim_out):
        super(GCN, self).__init__()
        self.gc1 = GCNConv(num_node_feats, 256)
        self.gc2 = GCNConv(256, 256)
        self.gc3 = GCNConv(256, 256)
        self.fc1 = nn.Linear(256, 196)
        self.fc2 = nn.Linear(196, dim_out)

    def forward(self, g):
        h = F.relu(self.gc1(g.x, g.edge_index))
        h = F.relu(self.gc2(h, g.edge_index))
        h = F.relu(self.gc3(h, g.edge_index))
        hg = global_mean_pool(h, g.batch)
        hg = F.relu(self.fc1(hg))
        out = self.fc2(hg)

        return out


# Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio,
# Graph Attention Networks,
# International Conference on Learning Representations (ICLR) 2018
class GAT(nn.Module):
    # graph attention network (GAT)
    def __init__(self, num_node_feats, dim_out):
        super(GAT, self).__init__()
        self.gc1 = GATConv(num_node_feats, 256)
        self.gc2 = GATConv(256, 256)
        self.gc3 = GATConv(256, 256)
        self.fc1 = nn.Linear(256, 196)
        self.fc2 = nn.Linear(196, dim_out)

    def forward(self, g):
        h = F.relu(self.gc1(g.x, g.edge_index))
        h = F.relu(self.gc2(h, g.edge_index))
        h = F.relu(self.gc3(h, g.edge_index))
        hg = global_mean_pool(h, g.batch)
        hg = F.relu(self.fc1(hg))
        out = self.fc2(hg)

        return out


# Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka,
# How Powerful are Graph Neural Networks?,
# International Conference on Learning Representations (ICLR) 2019
class GIN(nn.Module):
    # graph isomorphism network (GIN)
    def __init__(self, num_node_feats, dim_out):
        super(GIN, self).__init__()
        self.gc1 = GINConv(nn.Linear(num_node_feats, 256))
        self.gc2 = GINConv(nn.Linear(256, 256))
        self.gc3 = GINConv(nn.Linear(256, 256))
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, dim_out)

    def forward(self, g):
        h = F.relu(self.gc1(g.x, g.edge_index))
        h = F.relu(self.gc2(h, g.edge_index))
        h = F.relu(self.gc3(h, g.edge_index))
        hg = global_mean_pool(h, g.batch)
        h = F.relu(self.fc1(hg))
        out = self.fc2(h)

        return out


# Gyoung S. Na, Hyun Woo Kim, and Hyunju Chang,
# Costless Performance Improvement in Machine Learning for Graph-Based Molecular Analysis,
# J. Chem. Inf. Model, 2020, 60, 3, 1137-1145
class EGCN(nn.Module):
    # extended graph convolutional network
    def __init__(self, num_node_feats, dim_out):
        super(EGCN, self).__init__()
        self.gc1 = GCNConv(num_node_feats, 256)
        self.gc2 = GCNConv(256, 256)
        self.gc3 = GCNConv(256, 256)
        self.fc1 = nn.Linear(256 + 2, 196)
        self.fc2 = nn.Linear(196, dim_out)

    def forward(self, g):
        h = F.relu(self.gc1(g.x, g.edge_index))
        h = F.relu(self.gc2(h, g.edge_index))
        h = F.relu(self.gc3(h, g.edge_index))
        hg = global_mean_pool(h, g.batch)
        hg = torch.cat([hg, g.mol_wt, g.n_rings], dim=1)
        hg = F.relu(self.fc1(hg))
        out = self.fc2(hg)

        return out
