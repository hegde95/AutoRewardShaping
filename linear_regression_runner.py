from environments.MyFrozenLakeEnv import MyFrozenLake


import numpy as np
from tqdm import tqdm
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import imageio
from datetime import datetime

L = 16
M = 4
alpha = 0.456
# N = 30

log = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

os.makedirs("results", exist_ok=True)
absorbing_goal = False
save = True
if save:
    os.makedirs("results/"+log, exist_ok=True)


class STL():
    def __init__(self):
        self.h1_mid_state = L//2
        self.h2_mid_state = L//2        

    def get_dist(self, state1, state2):
        x1 = state2 % 4
        y1 = state2 // 4

        x2 = state1 % 4
        y2 = state1 // 4

        return abs(x1-x2) + abs(y1-y2)

    def get_min_hole_dist(self, state):

        # mid for hole 1
        min_dist3 = 9999
        dist_to_hole = self.get_dist(self.h1_mid_state, state)
        min_dist3 = min(min_dist3, dist_to_hole)

        # mid for hole 2
        dist_to_hole = self.get_dist(self.h2_mid_state, state)
        min_dist3 = min(min_dist3, dist_to_hole)

        return min_dist3

    def calc_costs(self,traj):
        min_dist_to_hole_mid = 9999
        
        for state in traj:
            step_cost_mid = self.get_min_hole_dist(state)
            min_dist_to_hole_mid = min(min_dist_to_hole_mid, step_cost_mid)
        return min_dist_to_hole_mid


losses = []
Ns = []
for N in range(12, 184):
    if N % 5 == 0:
        print(N)

        # State-Action visitation frequency
        K = np.zeros((N, L*M))

        # Next - State visitation frequency
        H = np.zeros((N, L))

        env = MyFrozenLake(probabilistic=False, absorbing_goal_state = absorbing_goal)
        C = []
        R = []
        Z = []
        trajs = []

        for j in range(N):
            # Step1 - Create a random trajectory
            # action_seq = list(np.random.choice([0,1,2,3], 30, p = [0.1,0.4,0.4,0.1]))
            action_seq = list(np.random.choice([0,1,2,3], 30))
            state_seq = []
            
            k = 0
            score = 0
            min_dist_to_hole = 9999

            st = env.reset()
            while k < len(action_seq):
                action = action_seq[k]
                k += 1
                ij_index = st * M + int(action)
                K[j,ij_index] += 1

                next_st, reward, done, info = env.step(action)
                H[j,next_st] += 1

                min_dist_to_hole = min(min_dist_to_hole, info["min_hole_dist"])

                # if next_st in env.constraint_states:
                score += reward
                state_seq.append(st)
                st = next_st
            
            n_score = np.random.normal(score, 0.5)
            R.append(n_score)
            # min_dist_to_hole = np.random.normal(min_dist_to_hole, 0.5)
            C.append(min_dist_to_hole)
            trajs.append(state_seq)
            z = score-alpha*min_dist_to_hole
            # n_z = np.random.normal(z, 0.5)
            Z.append(z)


            # if j%100 == 0:
            #     print("\n")
            #     print(j)

        R = np.array(R)
        C = np.array(C)
        Z = np.array(Z)

        #####################################################################################################################
        # Section to calc stl specifications
        #####################################################################################################################


        stl = STL()
        h1_upper = L
        h1_lower = 0
        h2_upper = L
        h2_lower = 0
        for j in range(10):
            
            # Get loss if h1 is upper limit, h2 is upper limit, 
            stl.h1_mid_state = h1_upper
            stl.h2_mid_state = h2_upper
            loss1 = 0
            for i, traj in enumerate(trajs):
                cost_actual = C[i]
                cost_mid_est = stl.calc_costs(traj)
                loss1 += (cost_mid_est-cost_actual)**2

            # Get loss if h1 is upper limit, h2 is lower limit, 
            stl.h1_mid_state = h1_upper
            stl.h2_mid_state = h2_lower
            loss2 = 0
            for i, traj in enumerate(trajs):
                cost_actual = C[i]
                cost_mid_est = stl.calc_costs(traj)
                loss2 += (cost_mid_est-cost_actual)**2

            # Get loss if h1 is lower limit, h2 is upper limit, 
            stl.h1_mid_state = h1_lower
            stl.h2_mid_state = h2_upper
            loss3 = 0
            for i, traj in enumerate(trajs):
                cost_actual = C[i]
                cost_mid_est = stl.calc_costs(traj)
                loss3 += (cost_mid_est-cost_actual)**2

            # Get loss if h1 is lower limit, h2 is lower limit, 
            stl.h1_mid_state = h1_lower
            stl.h2_mid_state = h2_lower
            loss4 = 0
            for i, traj in enumerate(trajs):
                cost_actual = C[i]
                cost_mid_est = stl.calc_costs(traj)
                loss4 += (cost_mid_est-cost_actual)**2


            if min([loss1, loss2, loss3, loss4]) == loss1:
                h1_lower = (h1_lower + h1_upper)//2
                h2_lower = (h2_lower + h2_upper)//2
                
            elif min([loss1, loss2, loss3, loss4]) == loss2:
                h1_lower = (h1_lower + h1_upper)//2
                h2_upper = (h2_lower + h2_upper)//2

            elif min([loss1, loss2, loss3, loss4]) == loss3:
                h1_upper = (h1_lower + h1_upper)//2
                h2_lower = (h2_lower + h2_upper)//2

            elif min([loss1, loss2, loss3, loss4]) == loss4:
                h1_upper = (h1_lower + h1_upper)//2
                h2_upper = (h2_lower + h2_upper)//2
            # print(min([loss1, loss2, loss3, loss4]))


        print("Holes are in:{} and {}".format((h1_upper + h1_lower)//2, (h2_upper + h2_lower)//2))

        #####################################################################################################################

        sym_mat1 = np.dot(K.transpose(), K)
        pseudo_inv1 = np.dot(np.linalg.inv(sym_mat1), K.transpose())
        r1 = np.dot(pseudo_inv1, R)

        sym_mat2 = np.dot(H.transpose(), H)
        pseudo_inv2 = np.dot(np.linalg.inv(sym_mat2), H.transpose())
        r2 = np.dot(pseudo_inv2, R)

        H_dash = np.concatenate((H,C.reshape(N,1)), axis = 1)
        sym_mat3 = np.dot(H_dash.transpose(), H_dash)
        pseudo_inv3 = np.dot(np.linalg.inv(sym_mat3), H_dash.transpose())
        r3 = np.dot(pseudo_inv3, Z) 

        print(r3[-1])       

        if save:
            mat1 = np.zeros((4,4))
            mat2 = np.zeros((4,4))
            for state in env.state_rewards:
                
                true_reward = env.state_rewards[state]
                predicted_reward = r2[state]



                mat1[state//4, state%4] = true_reward
                mat2[state//4, state%4] = predicted_reward

            fig2, (ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=3, figsize=(15, 10))

            ax3.title.set_text('reward true')
            plot1 = ax3.imshow(mat1, label='reward true', vmin = 0, vmax = 0.6)

            ax4.title.set_text('reward predicted')
            plot2 = ax4.imshow(mat2, label='reward predicted', vmin = 0, vmax = 0.6)

            ax5.title.set_text('error')
            plot3 = ax5.imshow(mat2-mat1, label='error', vmin = -0.5, vmax = 0.5)

            sum_squared_errors = np.sum((mat2-mat1)**2)
            losses.append(sum_squared_errors)
            Ns.append(N)

            fig2.colorbar(plot1,ax=ax3)
            fig2.colorbar(plot2,ax=ax4)
            fig2.colorbar(plot3,ax=ax5)

            fig2.suptitle("N = " + str(N) + "; error = " + str(sum_squared_errors))


            if absorbing_goal:
                fig2.savefig('results/'+ log +'/reward_matrix_'+str(N)+'_absorbing.png')
            else:
                fig2.savefig('results/'+ log +'/reward_matrix_'+str(N)+'_not_absorbing.png')


if save:
    sorted_file_names = {}
    filenames = os.listdir("results/"+log)

    for filename in filenames:
        N = int(filename.split("_")[2])
        sorted_file_names[N] = filename

    sorted_file_names = sorted_file_names.items()

    sorted_file_names = sorted(sorted_file_names)

    images = []
    for N, filename in sorted_file_names:
        images.append(imageio.imread("results/"+log+"/"+filename))
    imageio.mimsave('results/'+ log +'/rewards.gif', images)

    fig3, ax6 = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    ax6.plot(np.array(Ns),np.array(losses))
    ax6.set(xlabel='Number of trajectories', ylabel='Error')
    
    fig3.savefig('results/'+ log +'/loss_curve.png')

print("Done")