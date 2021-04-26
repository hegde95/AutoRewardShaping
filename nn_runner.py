from environments.MyFrozenLakeEnv import MyFrozenLake


import numpy as np
from tqdm import tqdm
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action])
        x = self.layers(x)
        return x

class MLP2(nn.Module):
    def __init__(self):
        super(MLP2, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )
        
    def forward(self, x):
        # x = torch.cat([state, action])
        x = self.layers(x)
        return x

L = 16
M = 4
N = 5000

absorbing_goal = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rew_func = MLP()
rew_func2 = MLP2()

optimizer = torch.optim.Adam(rew_func.parameters(), lr=0.0001)
optimizer2 = torch.optim.Adam(rew_func2.parameters(), lr=0.0001)
loss_fn = nn.MSELoss()
losses = []
losses2 = []

env = MyFrozenLake(probabilistic=False, absorbing_goal_state = absorbing_goal)
C = []
R = []

for j in range(N):
    # Step1 - Create a random trajectory
    # traj = list(np.random.choice([0,1,2,3], 30, p = [0.1,0.4,0.4,0.1]))
    traj = list(np.random.choice([0,1,2,3], 30))
    
    k = 0
    score = 0
    min_dist_to_hole = 9999

    pred_total_score_tensor = torch.tensor([0.0])
    pred_total_score_tensor2 = torch.tensor([0.0])

    st = env.reset()
    while k < len(traj):
        action = traj[k]
        k += 1
        
        next_st, reward, done, info = env.step(action)
        min_dist_to_hole = min(min_dist_to_hole, info["min_hole_dist"])

        # if next_st in env.constraint_states:
        score += reward


        state_tensor = torch.FloatTensor([st]).to(device)
        action_tensor = torch.FloatTensor([action]).to(device)
        next_state_tensor = torch.FloatTensor([next_st]).to(device)

        
        pred_reward_tensor = rew_func(state_tensor, action_tensor)
        pred_reward_tensor2 = rew_func2(next_state_tensor)
        # pred_reward = pred_reward_tensor.cpu().data.numpy().flatten()
        pred_total_score_tensor += pred_reward_tensor
        pred_total_score_tensor2 += pred_reward_tensor2

        st = next_st
    
    score = np.random.normal(score, 0.5)
    R.append(score)
    # min_dist_to_hole = np.random.normal(min_dist_to_hole, 0.5)
    C.append(min_dist_to_hole)

    real_total_score_tensor = torch.FloatTensor([score]).to(device)

    loss = loss_fn(pred_total_score_tensor, real_total_score_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.cpu().data.numpy().flatten())

    loss2 = loss_fn(pred_total_score_tensor2, real_total_score_tensor)
    loss2.backward()
    optimizer2.step()
    losses2.append(loss2.cpu().data.numpy().flatten())
    if j%100 == 0:
        print("\n")
        print(j)
        print(loss.cpu().data.numpy().flatten())

fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))


ax1.plot(losses, label='reward loss state-action')
ax1.legend()

ax2.plot(losses2, label='reward loss next-state')
ax2.legend()

fig1.savefig('reward_loss.png')


mat1 = np.zeros((4,4))
mat2 = np.zeros((4,4))
for state in env.state_rewards:
     
    true_reward = env.state_rewards[state]
    state_tensor = torch.FloatTensor([state]).to(device)
    predicted_reward = rew_func2(state_tensor)
    predicted_reward = predicted_reward.cpu().data.numpy().flatten()

    mat1[state//4, state%4] = true_reward
    mat2[state//4, state%4] = predicted_reward

fig2, (ax3, ax4) = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))
plot1 = ax3.imshow(mat1, label='reward true')

plot2 = ax4.imshow(mat2, label='predicted true')

fig2.colorbar(plot1,ax=ax3)
fig2.colorbar(plot2,ax=ax4)

if absorbing_goal:
    fig2.savefig('reward_matrix_absorbing.png')
else:
    fig2.savefig('reward_matrix_not_absorbing.png')

print("Done")