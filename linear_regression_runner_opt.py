from environments.MyFrozenLakeEnv import MyFrozenLake
from environments.MyFrozenLake8Env import MyFrozenLake8


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
import pyswarms as ps

L = 64
M = 4
alpha = 0.456
# N = 30
traj_len = 200
dim = int(np.sqrt(L))


log = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
base_dir = "results8"

os.makedirs(base_dir, exist_ok=True)
absorbing_goal = False
save = True
if save:
    os.makedirs(base_dir+"/"+log, exist_ok=True)


class STL():
    def __init__(self, num_holes):
        self.num_holes = num_holes

        self.holes = [0 for _ in range(self.num_holes)]      

    def get_dist(self, state1, state2):
        x1 = state2 % dim
        y1 = state2 // dim

        x2 = state1 % dim
        y2 = state1 // dim

        return abs(x1-x2) + abs(y1-y2)

    def get_min_hole_dist(self, state):

        # mid for hole 1
        min_dist = 9999

        for hole_est in self.holes:
                
            dist_to_hole = self.get_dist(hole_est, state)
            min_dist = min(min_dist, dist_to_hole)

        return min_dist

    def calc_costs(self,traj):
        min_dist_to_hole_mid = 9999
        
        for state in traj:
            step_cost_mid = self.get_min_hole_dist(state)
            min_dist_to_hole_mid = min(min_dist_to_hole_mid, step_cost_mid)
        return min_dist_to_hole_mid


losses = []
Ns = []
for N in range(14, 184):
# for N in range(14, 54):
    if N % 5 == 0:
        print(N)

        # State-Action visitation frequency
        K = np.zeros((N, L*M))

        # Next - State visitation frequency
        H = np.zeros((N, L))

        env = MyFrozenLake8(probabilistic=False, absorbing_goal_state = absorbing_goal)
        C = []
        R = []
        Z = []
        trajs = []

        for j in range(N):
            # Step1 - Create a random trajectory
            # action_seq = list(np.random.choice([0,1,2,3], 30, p = [0.1,0.4,0.4,0.1]))
            action_seq = list(np.random.choice([0,1,2,3], traj_len))
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

        h_dim = 2

        stl = STL(h_dim)
        def get_diff(X):
            n_particles = X.shape[0]
            n_holes = X.shape[1]

            dist = []
            for part in range(n_particles):
                for hole_n in range(n_holes):
                    stl.holes[hole_n] = X[part,hole_n]
                loss_part = 0
                for i, traj in enumerate(trajs):
                    cost_actual = C[i]
                    cost_mid_est = stl.calc_costs(traj)
                    loss_part += (cost_mid_est-cost_actual)**2
                dist.append(loss_part)
            return np.array(dist)

        # get_diff(np.array([[11,21,61,22,14,12,15,62,2,11,12,32,41,12,47,46],
        #                     [61,2,14,12,15,62,2,11,12,14,12,15,62,2,11,12],
        #                     [45,13,61,22,14,12,15,62,61,22,14,12,15,62,61,2]]))

        # get_diff(np.array([[11,21],
        #                     [61,2],
        #                     [45,13]]))

        swarm_size = 20
        constraints = (np.array([0, 0]),
               np.array([63, 63]))

        options = {'c1': 1.5, 'c2':1.5, 'w':0.5}


        optimizer = ps.single.GlobalBestPSO(n_particles=swarm_size,
                                            dimensions=h_dim,
                                            options=options,
                                            bounds=constraints)
        
        cost, hole_states = optimizer.optimize(get_diff, iters=25)
        
        
        print("Holes are in:{} and {}".format(int(hole_states[0]), int(hole_states[1])))

        #####################################################################################################################

        sym_mat1 = np.dot(K.transpose(), K)
        pseudo_inv1 = np.dot(np.linalg.inv(sym_mat1), K.transpose())
        r1 = np.dot(pseudo_inv1, R)

        sym_mat2 = np.dot(H.transpose(), H)
        pseudo_inv2 = np.dot(np.linalg.inv(sym_mat2), H.transpose())
        r2 = np.dot(pseudo_inv2, R)

        # H_dash = np.concatenate((H,C.reshape(N,1)), axis = 1)
        # sym_mat3 = np.dot(H_dash.transpose(), H_dash)
        # pseudo_inv3 = np.dot(np.linalg.inv(sym_mat3), H_dash.transpose())
        # r3 = np.dot(pseudo_inv3, Z) 

        # print(r3[-1])       

        if save:
            mat1 = np.zeros((dim,dim))
            mat2 = np.zeros((dim,dim))
            mat3 = np.zeros((dim,dim))
            mat4 = np.zeros((dim,dim))
            for state in env.state_rewards:
                
                true_reward = env.state_rewards[state]
                predicted_reward = r2[state]



                mat1[state//dim, state%dim] = true_reward
                mat2[state//dim, state%dim] = predicted_reward

                if state in [int(hole_states[0]), int(hole_states[1])]:
                    mat3[state//dim, state%dim] = 1

                if state in env.constraint_states:
                    mat4[state//dim, state%dim] = 1

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
                fig2.savefig(base_dir + '/'+ log +'/reward_matrix_'+str(N)+'_absorbing.png')
            else:
                fig2.savefig(base_dir + '/'+ log +'/reward_matrix_'+str(N)+'_not_absorbing.png')

            fig4, (ax7, ax8) = plt.subplots( nrows=1, ncols=2 )  # create figure & 1 axis
            ax7.title.set_text('hole est')
            ax7.imshow(mat3, label='hole est', vmin = 0, vmax = 0.6)

            ax8.title.set_text('hole real')
            ax8.imshow(mat4, label='hole real', vmin = 0, vmax = 0.6)
            fig4.suptitle("N = " + str(N))
            fig4.savefig(base_dir + '/'+ log +'/hole_matrix_'+str(N)+'.png')



if save:
    sorted_file_names = {}
    filenames = os.listdir(base_dir + "/"+log)

    for filename in filenames:
        if filename.split("_")[0] == "reward":
            N = int(filename.split("_")[2])
        
            sorted_file_names[N] = filename

    sorted_file_names = sorted_file_names.items()

    sorted_file_names = sorted(sorted_file_names)

    images = []
    for N, filename in sorted_file_names:
        images.append(imageio.imread(base_dir + "/"+log+"/"+filename))
    imageio.mimsave(base_dir + '/'+ log +'/rewards.gif', images)

    fig3, ax6 = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    ax6.plot(np.array(Ns),np.array(losses))
    ax6.set(xlabel='Number of trajectories', ylabel='Error')
    
    fig3.savefig(base_dir + '/'+ log +'/loss_curve.png')

    sorted_file_names = {}
    filenames = os.listdir(base_dir + "/"+log)

    for filename in filenames:
        if filename.split("_")[0] == "hole":
            N = int(filename.split("_")[2][:-4])
            sorted_file_names[N] = filename

    sorted_file_names = sorted_file_names.items()

    sorted_file_names = sorted(sorted_file_names)

    images = []
    for N, filename in sorted_file_names:
        images.append(imageio.imread(base_dir + "/"+log+"/"+filename))
    imageio.mimsave(base_dir + '/'+ log +'/holes.gif', images)

print("Done")