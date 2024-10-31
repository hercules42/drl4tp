


import numpy as np
import math
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from directedhgnn import DirectedHGNNet    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')


class Encoder(nn.Module):               
    """Encodes the static & dynamic states using 1d Convolution."""

    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__() 
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=1)           # 一层 的一维卷积

    def forward(self, input):
        output = self.conv(input)
        return output  # (batch, hidden_size, seq_len)


class Attention(nn.Module):
    """Calculates attention over the input nodes given the current state."""         # 计算attention

    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        # W processes features from static decoder elements
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size),
                                          device=device, requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 2* hidden_size),
                                          device=device, requires_grad=True))

    def forward(self, static_hidden, decoder_hidden):

        batch_size, _, hidden_size= static_hidden.size()

        hidden = decoder_hidden.expand_as(static_hidden)
        hidden = torch.cat((static_hidden,hidden), 2)     # （x,h)  x=(s,t)
        hidden = hidden.transpose(2,1)

        # Broadcast some dimensions so we can do batch-matrix-multiply
        v = self.v.expand(batch_size, 1, hidden_size)
        W = self.W.expand(batch_size, hidden_size, -1)

        attns = torch.bmm(v, torch.tanh(torch.bmm(W, hidden)))
        attns = F.softmax(attns, dim=2)  # (batch, seq_len)        #这两行是公式4
        return attns


class Pointer(nn.Module):
    """Calculates the next state given the previous state and input embeddings."""

    def __init__(self, hidden_size, num_layers=1, dropout=0.2):
        super(Pointer, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Used to calculate probability of selecting next state
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size),
                                          device=device, requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 3 * hidden_size),
                                          device=device, requires_grad=True))

        self.v_target = nn.Parameter(torch.zeros((1, 1, hidden_size),
                                         device=device, requires_grad=True))

        self.W_taret = nn.Parameter(torch.zeros((1, hidden_size, 3 * hidden_size),
                                          device=device, requires_grad=True))
        
  
        # Used to compute a representation of the current decoder output
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)  # 这里的GRU是用来计算decoder的输出的

  
        self.encoder_attn = Attention(hidden_size)
        self.traget_attn = Attention(hidden_size)
        self.drop_rnn = nn.Dropout(p=dropout)
        self.drop_hh = nn.Dropout(p=dropout)          #dropout防止过拟合

    def forward(self, static_hidden, state_embedding, decoder_hidden, target_hidden, last_hh):

        # print(self.v)
        #print("decoder_hidden",decoder_hidden)
        rnn_out, last_hh = self.gru(decoder_hidden, last_hh)
        rnn_out = self.drop_rnn(rnn_out)
        if self.num_layers == 1:
            last_hh = self.drop_hh(last_hh)         

        enc_attn = self.encoder_attn(static_hidden, rnn_out)
        context = enc_attn.bmm(static_hidden)  # (B, 1, num_feats)

        # Calculate the next output using Batch-matrix-multiply ops
        context = context.expand_as(static_hidden)
        state_embedding = state_embedding.expand_as(static_hidden)
        energy = torch.cat((static_hidden, context, state_embedding), dim=2)  # (B, num_feats*3, seq_len)

        energy = energy.transpose(2,1)
        v = self.v.expand(static_hidden.size(0), -1, -1)
        W = self.W.expand(static_hidden.size(0), -1, -1)

        probs = torch.bmm(v, torch.tanh(torch.bmm(W, energy))).squeeze(1)
        #print("encoder_attn",enc_attn)
        
        #从target计算probs
        enc_attn_target = self.traget_attn(static_hidden, target_hidden)
        context_target  = enc_attn_target.bmm(static_hidden)
        context_target = context_target.expand_as(static_hidden)

        energy_target = torch.cat((static_hidden, context_target, state_embedding), dim=2)

        energy_target = energy_target.transpose(2,1)
        v_target = self.v.expand(static_hidden.size(0), -1, -1)
        W_taret = self.W.expand(static_hidden.size(0), -1, -1)
        probs_target = torch.bmm(v_target, torch.tanh(torch.bmm(W_taret, energy_target))).squeeze(1)

        probs = 5*probs + probs_target # 不同权重
        # 2:1 -- 7.51279296875
        # 5:1 -- 7.4369140625
        
        return probs, last_hh


class DRL4TP(nn.Module):

    def __init__(self, static_size,hidden_size, update_fn=None, num_layers=1, dropout=0., seed=42):
        super(DRL4TP, self).__init__()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.update_fn = update_fn
        self.hidden_size = hidden_size
        # Define the encoder & decoder models

        self.static_encoder = Encoder(static_size, hidden_size)
        self.gnn_embedding = DirectedHGNNet(static_size+3, hidden_size)
        self.decoder = Encoder(hidden_size, hidden_size)
        self.pointer = Pointer(hidden_size, num_layers, dropout)

        h0 = torch.FloatTensor(hidden_size).to(device)
        self.h0 = nn.Parameter(h0)
        self.h0.data.uniform_(-1 / math.sqrt(hidden_size), 1 / math.sqrt(hidden_size))
        
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        
        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, static,weight, last_hh=None, actions=None):  ###


        batch_size, input_size, sequence_size = static.size()

        # 初始状态
        mask = torch.ones(batch_size, sequence_size, device=device)
        
        adj = torch.zeros(batch_size, sequence_size, sequence_size, device=device) #E*V
        
        
        tour_idx, tour_logp = [], []
        max_steps = 1000    

        # mask<--->state: 
        #  1: 没涉及到的点      [1,0,0]
        # 0: 被选中的点       [0,0,1]
        # 3: 被影响到但是没有发出信号（不在路径中）  [0,1,0]
        # -2: 上一步中被选择的点        [0,0,1]
        mask[:,0] = -2  # 初始状态选择第一个点
        mask_state = torch.zeros([batch_size,sequence_size,3], device=device) # [1,0,0]  [0,0,1] [0,1,0]

        state_flag1 = torch.tensor([1,0,0],dtype=torch.float,device=device)
        sate_flag2 = torch.tensor([0,0,1],dtype=torch.float,device=device)
        state_flag3 = torch.tensor([0,1,0],dtype=torch.float,device=device)

        mask_state[mask==1] = state_flag1
        mask_state[mask==0] = sate_flag2
        mask_state[mask==3] = state_flag3
        mask_state[mask==-2] = sate_flag2
        # 一开始选择第一个点
        
        state = torch.cat((static.transpose(2,1),mask_state), dim=-1)
        #static_embedding = self.static_encoder(static).transpose(2,1)

        node_embedding = self.gnn_embedding(state, adj)

        state_embedding = torch.mean(node_embedding, dim=1).unsqueeze(1)
        decoder_hidden = node_embedding[:,0,:].unsqueeze(1) #[batch_size, 1, n_features]
        target_hidden = node_embedding[:,-1,:].unsqueeze(1)
 

        if last_hh == None:
            h0 = self.h0.unsqueeze(0).expand(batch_size, self.hidden_size)
            h0 = h0.unsqueeze(0).contiguous()
            #print("node",node_embedding.shape)
            _, last_hh = self.gru(node_embedding, h0)  
            #print("last_hh",last_hh.shape)
            
        for step_idx in range(max_steps):

            check  = mask[:,-1:]   
            #print("check",check.squeeze(1))
            if check.ne(1).byte().all() and check.ne(3).byte().all():  # 如果全不为1 则训练完成   
                #print('break')
                break

            probs_pre, last_hh = self.pointer(node_embedding,state_embedding, decoder_hidden, target_hidden, last_hh)
            probs = torch.where(mask<=0,torch.tensor(-1*np.inf,device=device,dtype=torch.float),probs_pre)
            '''
            log = mask.log()
            log = torch.where(log>0, torch.tensor(0).to(device), log) # 对于mask=3的情况
            log = torch.where(torch.isnan(log),torch.tensor(-1*np.inf).to(device),log) # 对于mask=nan的情况, log) # 对于mask<0的情况
            #print("probs",probs)
            probs = F.softmax(probs + log, dim=1)     # 这里对mask取log，所以mask不能取负数 (把nan 换成 -inf)
            #print("log",log)
            #print(probs)
            '''
            probs1 = probs.clone()
            flag = (mask[:,-1] <= 0) # 上一轮被选中 后面就一直选这个点
            probs1[flag,-1] = 1
            probs1[flag,:-1] = 0
            probs = probs1.clone()

            if self.training:
                #print(probs)
                try:
                    m = torch.distributions.Categorical(probs)   # 不能让probs中某行的每个元素都为 -inf
                except:
                    print(probs_pre[:,:6])
                    print(probs_pre[:,8:])
                    #print(probs)
                    print(step_idx)
                    
                ptr = m.sample()
                logp = m.log_prob(ptr) #用categorical的log_prob计算logp得到的不是直接logp，而是log(softmax(probs)）

            else:
                prob, ptr = torch.max(probs, 1)  # Greedy
                logp = prob.log()

            if self.update_fn is not None:
                adj, mask, chosen_idx = self.update_fn(mask, adj, ptr.data, weight,self.training)       # update mask  ###
                adj = adj.detach()
                mask = mask.detach()
   

            tour_logp.append(logp.unsqueeze(1))
            tour_idx.append(chosen_idx.unsqueeze(1))

            decoder_hidden = torch.gather(node_embedding, 1,
                                         ptr.view(-1, 1, 1)
                                         .expand(-1, node_embedding.shape[-1], 1)).detach().transpose(2,1)

            mask_state[mask==1] = state_flag1
            mask_state[mask==0] = sate_flag2
            mask_state[mask==3] = state_flag3
            mask_state[mask==-2] = sate_flag2
            state = torch.cat((static.transpose(2,1),mask_state), dim=-1)
            node_embedding = self.gnn_embedding(state, adj)
            state_embedding = torch.mean(state_embedding, dim=1).unsqueeze(1)
       
        tour_idx = torch.cat(tour_idx, dim=1)  # (batch_size, seq_len)
        tour_logp = torch.cat(tour_logp, dim=1)  # (batch_size, seq_len)

        return tour_idx, tour_logp, mask   ##


if __name__ == '__main__':
    raise Exception('Cannot be called from main')
