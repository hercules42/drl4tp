"""

与0407版的修改：
0. 设置动态调整actor的学习率
1. x0设置为第一个点
2. h0由输入为整个图数据的rnn网络得到，而非随机生成
3. 提升训练速度  （update_mask部分不用循环，光改了这个就提高了十倍！！）
3.5 删除dynamic的部分

3.9 用greedy算法取代critic
4. 多头注意力（两个注意力模块的权重不一样）
test on 10000: 20 -- 7.51(两个注意力模块比例2：1)
                    xx(两个注意力模块比例5：1)
               50 -- 12.68 (critic)

#TODO

5.  修改点的表示
6.  用hgnn
7. 用除了actor-critic的其他算法
8. 每个step时还应考虑影响的是新增的点还是已有的点
9. 用求解器求最优解，和现在的对比一下
10. attention中不经过context，注意力分数作为概率；
    或者其他方法求context，因为目前context并没有考虑当前步之前选择的点 (可以用dynamic来表示点的状态，作为context)

"""

import os
import time
import argparse
import datetime
import numpy as np
from tasks import tp
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from model_state_hg_onehot_target import DRL4TP, Encoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')


class StateCritic(nn.Module):
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, static_size, hidden_size):
        super(StateCritic, self).__init__()

        self.static_encoder = Encoder(static_size, hidden_size)
        # Define the encoder & decoder models
        self.fc1 = nn.Conv1d(hidden_size , 20, kernel_size=1)
        self.fc2 = nn.Conv1d(20, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static):

        # Use the probabilities of visiting each
        static_hidden = self.static_encoder(static)

        hidden = static_hidden

        output = F.relu(self.fc1(hidden))
        output = F.relu(self.fc2(output))
        output = self.fc3(output).sum(dim=2)
        return output


class Critic(nn.Module):
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, hidden_size):
        super(Critic, self).__init__()

        # Define the encoder & decoder models
        self.fc1 = nn.Conv1d(1, hidden_size, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden_size, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input):

        output = F.relu(self.fc1(input.unsqueeze(1)))
        output = F.relu(self.fc2(output)).squeeze(2)
        output = self.fc3(output).sum(dim=2)
        return output


def validate(data_loader, actor, reward_fn, render_fn=None, save_dir='.',
             num_plot=5):
    """Used to monitor progress on a validation set & optionally plot solution."""

    actor.eval()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rewards = []
    for batch_idx, batch in enumerate(data_loader):

        static, x0, xt = batch

        weight = tp.weight_fn(static)            ###
        neighbour_mask = (weight>0) & (weight < 10)  # 就是最近的五个 因为要排除自己
        adj = weight.clone()
        adj[neighbour_mask] = 1
        adj[~neighbour_mask] = 0
        adj = adj.transpose(1,2)

        with torch.no_grad():
            tour_indices, _ , mask = actor.forward(static, adj, weight, x0, xt)  ###

        tour_indices = torch.cat((torch.zeros(static.size(0),1,device=device,dtype=int), tour_indices), dim=1)
        reward = reward_fn(mask)
        reward = reward.float()
        reward = reward.mean().item()
        #  reward = reward_fn(mask).mean().item()
        rewards.append(reward)

        if render_fn is not None and batch_idx < num_plot:
            name = 'batch%d_%2.4f.png'%(batch_idx, reward)
            path = os.path.join(save_dir, name)
            render_fn(static, tour_indices, mask, path)

    actor.train()
    return np.mean(rewards)


def train(actor, critic, task, num_nodes, train_data, valid_data, reward_fn,
          render_fn, batch_size, actor_lr, critic_lr, max_grad_norm,
          **kwargs):
    """Constructs the main actor & critic networks, and performs all training."""

    now = '%s' % datetime.datetime.now().time()
    now = now.replace(':', '_')
    save_dir = os.path.join(task, 'ppo_%d' % num_nodes, now)

    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)

    lr_decay_step = 3907  # 相当于每个epoch更新一次
    lr_decay_rate = 0.85
    actor_scheduler = lr_scheduler.StepLR(actor_optim, lr_decay_step, lr_decay_rate)
    critic_scheduler = lr_scheduler.StepLR(critic_optim, lr_decay_step, lr_decay_rate)
    train_loader = DataLoader(train_data, batch_size, True, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size, False, num_workers=0)

    best_params = None
    best_reward = np.inf
    epsilon = 0.2   

    epochs_ppo = 10

    for epoch in range(15):                  # 本来是10

        actor.train()
        critic.train()

        times, losses, rewards, critic_rewards = [], [], [], []

        epoch_start = time.time()
        start = epoch_start

        for batch_idx, batch in enumerate(train_loader):

            static, x0, xt = batch
         
            weight = tp.weight_fn(static)


            # Full forward pass through the dataset
            with torch.no_grad():

                tour_indices, old_tour_logp, mask = actor(static,weight)  ###
                tour_indices = torch.cat((torch.zeros(static.size(0),1,device=device,dtype=int), tour_indices), dim=1)
                #print(tour_logp[0],tour_indices[0])
                #print("............................")
                #tour_indices2, tour_logp2, mask2 = actor(static,adj,weight, x0, xt, actions=tour_indices)  ###
                #print(tour_logp2[0],tour_indices2[0])
                #print("tour",tour_logp[0],tour_indices[0])

                #print("tour2",tour_logp2[0],tour_indices2[0])
                reward = reward_fn(mask).float()
                #print("采样的概率：",old_tour_logp[0])
                old_logp = old_tour_logp.sum(dim=1)
                
                critic_est = critic(static).view(-1)
                advantage = (reward - critic_est)
                advantage = (advantage-advantage.mean())/(advantage.std()+1e-5)
                #print(advantage)

            for ep_idx in range(epochs_ppo):

                tour_indices, tour_logp, mask = actor(static,weight)
                #if ep_idx <=1:
                #    print(tour_logp[0])

                tour_logp = tour_logp.sum(dim=1)
                ratio = torch.exp(tour_logp - old_logp)

                # if ep_idx <=1:
                #     print("ratio_step:",ratio_step[0])
                #     print("ratio:",ratio)
                
                surr1 = ratio * advantage.detach()
                surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantage.detach()
                actor_loss = torch.min(surr1, surr2).mean()
                #if batch_idx >200:
                  #  print(actor_loss)
                #print("ep_idx:", ep_idx,tour_logp - old_logp)

                actor_optim.zero_grad() 
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                actor_optim.step()
                actor_scheduler.step()
        
                critic_est = critic(static).view(-1)
                critic_loss = torch.mean((reward - critic_est) ** 2)
                critic_optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
                critic_optim.step()
                critic_scheduler.step

                critic_rewards.append(torch.mean(critic_est.detach()).item())
                rewards.append(torch.mean(reward.detach()).item())
                losses.append(torch.mean(actor_loss.detach()).item())

            if (batch_idx + 1) % 1000 == 0:
                end = time.time()
                times.append(end - start)
                start = end

                mean_loss = np.mean(losses[-1000:])
                mean_reward = np.mean(rewards[-1000:])

                print('  Batch %d/%d, reward: %2.3f, loss: %2.4f, took: %2.4fs' %
                      (batch_idx, len(train_loader), mean_reward, mean_loss,
                       times[-1]))

        mean_loss = np.mean(losses)
        mean_reward = np.mean(rewards)

        # Save the weights
        epoch_dir = os.path.join(checkpoint_dir, '%s' % epoch)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        save_path = os.path.join(epoch_dir, 'actor.pt')
        torch.save(actor.state_dict(), save_path)

        save_path = os.path.join(epoch_dir, 'critic.pt')
        torch.save(critic.state_dict(), save_path)

        # Save rendering of validation set tours
        valid_dir = os.path.join(save_dir, '%s' % (epoch))
        mean_valid = validate(valid_loader, actor, reward_fn, render_fn,
                              valid_dir, num_plot=5)
        
        
        # Save best model parameters
        if mean_valid < best_reward:

            best_reward = mean_valid

            save_path = os.path.join(save_dir, 'actor.pt')
            torch.save(actor.state_dict(), save_path)

            save_path = os.path.join(save_dir, 'critic.pt')
            torch.save(critic.state_dict(), save_path)

        write_array_to_file(rewards, os.path.join(save_dir, 'rewards.txt'))
        write_array_to_file([mean_valid], os.path.join(save_dir, 'valid_perEpoch.txt'))
        print('Mean epoch %d loss/reward: %2.4f, %2.4f, %2.4f, took: %2.4fs '\
              '(%2.4fs / 100 batches)\n' % \
              (epoch, mean_loss, mean_reward, mean_valid, time.time() - epoch_start,
              np.mean(times)))

def write_array_to_file(array, file_name):
    # 检查文件是否存在
    if not os.path.exists(file_name):
        # 如果文件不存在，创建一个新的文件
        with open(file_name, 'a') as file:
            pass
        # 打开文件并写入数组内容
    with open(file_name, 'a') as file:
        for item in array:
            file.write(f"{item}\n") 

def train_tp(args):

    from tasks import tp
    from tasks.tp import TPDataset

    STATIC_SIZE = 2 # (x, y)

    train_data = TPDataset(args.num_nodes, args.train_size, args.seed)
    valid_data = TPDataset(args.num_nodes, args.valid_size, args.seed + 1)
    

    actor = DRL4TP(STATIC_SIZE,
                    args.hidden_size,
                    tp.update_mask,
                    args.num_layers,
                    args.dropout, args.seed).to(device)

    critic = StateCritic(STATIC_SIZE, args.hidden_size).to(device)

    kwargs = vars(args)
    kwargs['train_data'] = train_data
    kwargs['valid_data'] = valid_data
    kwargs['reward_fn'] = tp.reward  #不加括号，指调用函数体本身
    kwargs['render_fn'] = tp.render
    # kwargs['weight'] = TPDataset.weight

    if args.checkpoint:
        path = os.path.join(args.checkpoint, 'actor.pt')
        actor.load_state_dict(torch.load(path, device))

        path = os.path.join(args.checkpoint, 'critic.pt')
        critic.load_state_dict(torch.load(path, device))

    if not args.test:
        train(actor, critic, **kwargs)

    test_data = TPDataset(args.num_nodes, 10000, args.seed + 2)

    test_dir = 'test'
    test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)
    out = validate(test_loader, actor, tp.reward, tp.render, test_dir, num_plot=5)

    print('Average tour length: ', out)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Combinatorial Optimization')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--task', default='tp')
    parser.add_argument('--nodes', dest='num_nodes', default=20, type=int)
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--critic_lr', default=5e-4, type=float)
    parser.add_argument('--max_grad_norm', default=2., type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden', dest='hidden_size', default=128, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--layers', dest='num_layers', default=1, type=int)
    parser.add_argument('--train-size',default=1000000, type=int)     ## 本来是100万
    parser.add_argument('--valid-size', default=10000, type=int)

    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    #print('NOTE: SETTTING CHECKPOINT: ')
    #args.checkpoint = os.path.join('vrp', '10', '12_59_47.350165' + os.path.sep)
    #print(args.checkpoint)

    if args.task == 'tp':
        train_tp(args)
    else:
        raise ValueError('Task <%s> not understood'%args.task)

"""
run like
python trainer.py --task=tp --node=6
"""
