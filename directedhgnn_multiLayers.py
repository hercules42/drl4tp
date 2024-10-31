import torch
import torch.nn as nn
import torch.nn.functional as F


#层数怎么调
#dropout怎么调
class DirectedHGNNet(nn.Module):
    def __init__(self,n_obs_in,n_features, n_layers=3):

        super().__init__()
        self.init_node_embedding = nn.Linear(n_obs_in, n_features, bias=True)
        self.v2v = nn.ModuleList([vertex2vertex(n_features) for _ in range(n_layers)])
        self.n_layers = n_layers
        
    def forward(self, node_features, adj):

        # 先假设adj中起点是1，终点是-1 
        # adj的形式应该是E*V
        v_embeddings = self.init_node_embedding(node_features)
        for i in range(self.n_layers):
            v_embeddings = self.v2v[i](v_embeddings, adj)

        #print("init_node_embedding.weight: ",self.init_node_embedding.weight)
        return v_embeddings

class vertex2vertex(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.v2he = vertex2hyperedge(n_features)
        self.he2v = hyperedge2vertex(n_features)

    def forward(self, node_embeddings, adj):

        he_embeddings = self.v2he(node_embeddings, adj)
        v_embeddings = self.he2v(node_embeddings, he_embeddings, adj)
        return v_embeddings
    
class vertex2hyperedge(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.v2e_layer = nn.Linear(n_features, n_features, bias=True)

 
    def forward(self,node_embeddings, adj):
        #print("v2e_layer.weight: ",self.v2e_layer.weight)
        adj_target = torch.zeros_like(adj) 
        adj_target[adj==-1] = 1

        norm = torch.sum(adj_target,dim=-1).unsqueeze(-1)
        norm[norm==0] = 1   
        
        message_vertex = torch.matmul(adj_target, node_embeddings) / norm
        hyperedge_embeddings = F.relu(self.v2e_layer(message_vertex))
        return hyperedge_embeddings
    
class hyperedge2vertex(nn.Module):

    """
    超边到起点的消息传递，不需要NORM操作, 因为起点只有一个
    """
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.e2v_layer = nn.Linear(2*n_features, n_features, bias=True)

    def forward(self, vertex_embeddings, hyperedge_embeddings, adj):

        adj_T = adj.transpose(1,2)
        adj_T_source = torch.zeros_like(adj_T)
        adj_T_source[adj_T==1] = 1
        message_hyperedge = torch.matmul(adj_T_source, hyperedge_embeddings)
        vertex_embeddings = F.relu(self.e2v_layer(torch.cat([vertex_embeddings, message_hyperedge], dim=-1)))
        return vertex_embeddings
  
       