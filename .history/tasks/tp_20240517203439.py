"""Defines the main task for the tp
考虑终点

"""

import os
import numpy as np
import torch
import math
from torch.utils.data import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TPDataset(Dataset):                                                         # 继承torch的Dataset类

    def __init__(self, size=50, num_samples=1e6, seed=None):                        
        super(TPDataset, self).__init__()

        if seed is None:
            seed = np.random.randint(1222)
            
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.dataset = torch.rand((num_samples, 2, size)).to(device)          # size 行 2 列  的num_samples个样本
        self.num_nodes = size     # 每张图中点的个数
        self.size = num_samples     # 图的个数   
        self.x0 = self.dataset[:,:,:1]    
        self.xt = self.dataset[:,:,-1:] 
        
   


    def __len__(self):
        return self.size

    def __getitem__(self, idx):         # 第idx个图
        # (static, dynamic, start_loc)
        return (self.dataset[idx],  self.x0[idx], self.xt[idx])

"""
    tp 问题分三种点 
    1. 未被影响过
    -1. 被影响到但是没有发出信号（不在路径中）
    0. 被选中的点
    每次更新update的时候,不是一个点一个点的更新,每次选中一个新的点,会有一组点的mask被改变,
    这组点是上个点的影响半径（上个点到这个点的距离）内的点的集合
"""

def weight_fn(static):       # dataset[i][0][j] 表示第i张图的第j个点的横坐标
    # print(static[0])
    
    size, _, num_nodes = static.size()


    static_permute = static.permute(0, 2, 1)
    B, N, _ = static_permute.shape

    dist = -2 * torch.matmul(static_permute, static) #permute为转置,[B, N, M]
    dist += torch.sum(static_permute ** 2, -1).view(B, N, 1) #[B, N, M] + [B, N, 1]，dist的每一列都加上后面的列值
    dist += torch.sum(static_permute ** 2, -1).view(B, 1, N) #[B, N, M] + [B, 1, M],dist的每一行都加上后面的行值
    dist = torch.sqrt(dist)
    dist = torch.where(torch.isnan(dist), torch.full_like(dist, 0), dist)  # dist [B, N, N]

    # dist_mean = dist.mean(dim=-1)   # [batch_size,num_nodes]
    #weight = dist.sort()[1]
    #weight = weight.sort()[1]

    return dist


def update_mask(mask, adj, chosen_idx, weight, is_training=True):  ###
 
    # mask 这一步被访问的节点
    # chosen_idx (batch_size) 一维向量
    # weight 的 size : (batch_size, num_cities, num_cities) weight[i][j][k] 表示第i张图中第j个点影响到的第k个点 , 要注意一下起点的特例
    # 上一个点在mask的标记中是 -2 
    # 取出mask = -2 的点，赋给 last_chosen_idx
    weight_nodes = weight
    if mask.ne(-2).byte().all():    # 没有last_chosen_idx
        mask.scatter_(1, chosen_idx.unsqueeze(1), -2)
    else :
        last_chosen_idx = torch.nonzero(mask == -2)[:,-1:].squeeze(1)  # shape [batch_size]
        mask.scatter_(1, last_chosen_idx.unsqueeze(1), 0)    # 把上一轮的点还原成0
        
        if False:
            #如果这次选的点之前已经是被影响到的点 也是不对的
            chosen_weights = weight_nodes[np.arange(len(chosen_idx)), last_chosen_idx, chosen_idx] # 这次选择的点在上一步的点中的权重
            target_weights = weight_nodes[np.arange(len(chosen_idx)), last_chosen_idx, -1] # 终点在上一步的权重
            #如果终点是窃听的点，那么把终点置为该超边的目标结点     
            is_target = target_weights < chosen_weights   
            chosen_idx[is_target] = weight_nodes.size(2)-1   # 终点的索引是weight_nodes.size(2)-1
            #更新chosen_idx之后再更新mask
            #选择的点也要变
        
        mask.scatter_(1, chosen_idx.unsqueeze(1), -2) #被选中的点 设置为-2

        chosen_weights = weight_nodes[np.arange(len(chosen_idx)), last_chosen_idx, chosen_idx]
        
        last_chosen_weights = weight_nodes[np.arange(len(last_chosen_idx)), last_chosen_idx]  # 提取上一个点对各个点的权重
        
        hear = last_chosen_weights <= chosen_weights.unsqueeze(1) #hear的shape为[batch_size,num_cities]
        update_mask = hear & (mask == 1)
        mask[update_mask] = 3

        #mpnn时的adj
        #adj[np.arange(len(chosen_idx)), last_chosen_idx] = hear.float()
        #hg时的adj
        adj[np.arange(len(chosen_idx)), last_chosen_idx] =  -1*hear.float() #终点的adj值为-1
        adj[np.arange(len(chosen_idx)), last_chosen_idx, last_chosen_idx] = 1 #起点的adj值为1

        return adj, mask, chosen_idx


def reward(mask):

    return mask.ne(1).sum(1)   #被影响过的点的数量    #越小越好


def render(static, tour_indices, mask, save_path):
    
    """Plots the found tours."""

    # tour_indices: torch.IntTensor of size (batch_size, num_cities)
    # 在 tp 里， 不一定要走过所有点
    plt.close('all')

    # print("\nstatic")
    # print(static)

    num_plots = 3 if int(np.sqrt(len(tour_indices))) >= 3 else 1

    _, axes = plt.subplots(nrows=num_plots, ncols=num_plots,
                           sharex='col', sharey='row')

    if num_plots == 1:
        axes = [[axes]]
    axes = [a for ax in axes for a in ax]

    for i, ax in enumerate(axes):

        # Convert the indices back into a tour
        idx = tour_indices[i]
        if len(idx.size()) == 1:
            idx = idx.unsqueeze(0)      # 把idx从一维向量变成 （1，num_sequence)的二维向量



        """
        分成几类？
        1. 起点 终点        --> start, end
        2. 在路径上的点     --> tour_indices --> data
        3. 被影响到的点     --> eavesdropper
        4. 其他点     mask = 1
        1 应该包含在2 3 中
        """
        # End tour at the starting index
        idx = idx.expand(static.size(1), -1)    #把idx 变成size 为 (2,num_sequence) 的张量
        # idx = torch.cat((idx, idx[:, 0:1]), dim=1)
        mask_idx = mask[i]  

        idx_eavesdropper = [t for t in range(len(mask_idx)) if mask_idx[t] == 3 ]
        idx_eavesdropper = torch.tensor(idx_eavesdropper)
        idx_eavesdropper = idx_eavesdropper.expand(static.size(1), -1)
        num_sequence = len(mask_idx)
        start = torch.tensor([[0],[0]])
        end = torch.tensor([[num_sequence-1],[num_sequence-1]])

        data_all = static[i].data.cpu().numpy()    # 所有点
        data = torch.gather(static[i].data, 1, idx).cpu().numpy()   # 轨迹
        data_start = torch.gather(static[i].data, 1, start.to(device)).cpu().numpy()
        data_end = torch.gather(static[i].data, 1, end.to(device)).cpu().numpy()
        data_eavesdropper = torch.gather(static[i].data, 1, idx_eavesdropper.to(device)).cpu().numpy()

        #plt.subplot(num_plots, num_plots, i + 1)
        ax.plot(data[0], data[1], zorder=1)    # 画线
        ax.scatter(data_all[0], data_all[1], s=10, c='k', zorder=2)
        ax.scatter(data[0], data[1], s=10, c='r', zorder=6)
        ax.scatter(data_eavesdropper[0], data_eavesdropper[1], s=10, c='y', zorder=5) # 黄色表示被影响到的点
        ax.scatter(data_start[0], data_start[1], s=40, c='k', marker='^', zorder=3)    #起点用正三角形表示
        ax.scatter(data_end[0], data_end[1], s=40, c='k', marker='v', zorder=4)    #终点用倒三角形表示


        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect(1)
    plt.tight_layout()

    plt.savefig(save_path, bbox_inches='tight', dpi=400)


