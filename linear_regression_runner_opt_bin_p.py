import random 
import numpy as np

random.seed(1234)
np.random.seed(1234)

from environments.MyFrozenLakeEnv import MyFrozenLake
from environments.MyFrozenLake8Env import MyFrozenLake8


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
# import InverseProblem.functions as ip
from utils.functions import invert




L = 64
M = 4
alpha = 0.456
# N = 30
traj_len = 100
dim = int(np.sqrt(L))
reg_lambda = 10


log = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
base_dir = "results8_bin"

os.makedirs(base_dir, exist_ok=True)
absorbing_goal = False


save = False
if save:
    os.makedirs(base_dir+"/"+log, exist_ok=True)


class STL1():
    def __init__(self, num_holes):
        self.num_holes = num_holes

        self.holes = [0 for _ in range(self.num_holes)]      

    def set_holes(self, holes):
        self.holes = holes

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

class STL2():
    def __init__(self, num_holes):
        self.num_holes = num_holes

        # self.holes = [0 for _ in range(self.num_holes)] 
        
    
    def set_holes(self, holes):
        self.holes = holes
        self.dist_mat  = self.get_min_dist_matrix()


    def calc_costs(self,H_vec):
        H_vec = np.clip(H_vec,0,1)
        return min([state*self.dist_mat[inde] for inde, state in enumerate(H_vec) if state])

    def get_min_dist_matrix(self):
        mat = np.zeros(L)
        # mat = np.zeros((dim, dim))
        for state in range(L):
            mat[state] = self.get_min_hole_dist(state)
        return mat

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


losses = []
Ns = []
alpha_caps = []
for N in range(14, 184):
# for N in range(14, 54):
    if N % 5 == 0:
        print("Calculating for {} trajectories".format(N))

        # State-Action visitation frequency
        K = np.zeros((N, L*M))

        # Next - State visitation frequency
        H = np.zeros((N, L))

        env = MyFrozenLake8(probabilistic=False, absorbing_goal_state = absorbing_goal)
        C = []
        R = []
        Z = []
        # trajs = []

        for j in range(N):
            # Step1 - Create a random trajectory
            # action_seq = list(np.random.choice([0,1,2,3], 30, p = [0.1,0.4,0.4,0.1]))
            action_seq = list(np.random.choice([0,1,2,3], traj_len))
            state_seq = []
            
            k = 0
            score = 0
            min_dist_to_hole = 9999

            st = env.reset()
            info = env.get_info()
            min_dist_to_hole = min(min_dist_to_hole, info["min_hole_dist"])
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
            # trajs.append(state_seq)
            z = score-alpha*min_dist_to_hole
            # n_z = np.random.normal(z, 0.5)
            Z.append(z)

        R = np.array(R)
        C = np.array(C)
        Z = np.array(Z)

        #####################################################################################################################
        # Section to calc stl specifications
        #####################################################################################################################

        h_dim = 2

        # stl1 = STL1(h_dim)
        stl2 = STL2(h_dim)

        # Create a power set first
        power_set = []
        for a in range(1,L):
            for b in range(a):
                # power_set.append((a,b))
                x_i = np.zeros(L)
                x_i[a] = 1
                x_i[b] = 1
                power_set.append(x_i)
            
        # Iterate over the power set
        compliant_set = {} #dictionary with keys as objective and values as elemets of feasible set that satisfies constraints
        for hole_est in power_set:

            # Convert hole from indicator to list of states
            hole_index = 0
            holes = [0 for _ in range(h_dim)]
            for hole_n, hole_bool in enumerate(hole_est):
                if hole_bool:
                    holes[hole_index] = hole_n
                    hole_index += 1

            
            # stl1.set_holes(holes)
            stl2.set_holes(holes)
            
            check = True
            sum_obj = 0

            # Iterate over the N trajectories, each of them being a constraint, if the hole estimate satifies the constraint then add it to the dictionary
            # for i, traj in enumerate(trajs):
            # for i, H_i in enumerate(H):
            for i in range(N):
                cost_actual = C[i]
                cost_mid_est = stl2.calc_costs(H[i])
                # cost_mid_est1 = stl1.calc_costs(trajs[i])
                sum_obj += cost_mid_est
                if not (cost_actual <= cost_mid_est):
                    check = False
            if check:
                if sum_obj in compliant_set.keys():
                    compliant_set[sum_obj].append(hole_est)
                else:
                    compliant_set[sum_obj] = [hole_est]
                # compliant_set.append((sum_obj, hole_est))

        # Find minimum over all sums in the dictionary to find the answer
        hole_states_tup = compliant_set[min(compliant_set.keys())]

        # Can have multiple solutions so iterate over them to find the final set
        hole_states = []
        for x_i in hole_states_tup:
            h_1, h_2 = np.argwhere(x_i == np.amax(x_i))
            if h_1 not in hole_states:
                hole_states.append(h_1[0])
            if h_2 not in hole_states:
                hole_states.append(h_2[0])
        # print("Holes are in:{} and {}".format(int(hole_states[0]), int(hole_states[1])))
        print(hole_states)
        if (35 not in hole_states) or (20 not in hole_states):
            print("Something wrong here")
        #####################################################################################################################

        # sym_mat1 = np.dot(K.transpose(), K)
        # pseudo_inv1 = np.dot(np.linalg.inv(sym_mat1), K.transpose())
        # r1 = np.dot(pseudo_inv1, R)
        _,r1 =  invert(K, R, 10, 1)

        # sym_mat2 = np.dot(H.transpose(), H)
        # pseudo_inv2 = np.dot(np.linalg.inv(sym_mat2), H.transpose())
        # r2 = np.dot(pseudo_inv2, R)
        _,r2 =  invert(H, R, 10, 1)

        H_dash = np.concatenate((H,C.reshape(N,1)), axis = 1)
        # sym_mat3 = np.dot(H_dash.transpose(), H_dash)
        # pseudo_inv3 = np.dot(np.linalg.inv(sym_mat3), H_dash.transpose())
        # r3 = np.dot(pseudo_inv3, Z)
        _,r3 =  invert(H_dash, Z, 10, 1)

        print(r3[-1]) 
        alpha_caps.append(r3[-1])     

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

                if state in hole_states:
                    mat3[state//dim, state%dim] = 1

                if state in env.constraint_states:
                    mat4[state//dim, state%dim] = 1

            plt.close('all')
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 10))

            ax1.title.set_text('reward true')
            plot1 = ax1.imshow(mat1, label='reward true', vmin = 0, vmax = 0.6)

            ax2.title.set_text('reward predicted')
            plot2 = ax2.imshow(mat2, label='reward predicted', vmin = 0, vmax = 0.6)

            ax3.title.set_text('error')
            plot3 = ax3.imshow(mat2-mat1, label='error', vmin = -0.5, vmax = 0.5)

            sum_squared_errors = np.sum((mat2-mat1)**2)
            losses.append(sum_squared_errors)
            Ns.append(N)

            fig.colorbar(plot1,ax=ax1)
            fig.colorbar(plot2,ax=ax2)
            fig.colorbar(plot3,ax=ax3)

            fig.suptitle("N = " + str(N) + "; error = " + str(sum_squared_errors))


            if absorbing_goal:
                fig.savefig(base_dir + '/'+ log +'/reward_matrix_'+str(N)+'_absorbing.png')
            else:
                fig.savefig(base_dir + '/'+ log +'/reward_matrix_'+str(N)+'_not_absorbing.png')

            plt.close('all')
            fig, (ax1, ax2) = plt.subplots( nrows=1, ncols=2 )  # create figure & 1 axis
            ax1.title.set_text('hole est')
            ax1.imshow(mat3, label='hole est', vmin = 0, vmax = 0.6)

            ax2.title.set_text('hole real')
            ax2.imshow(mat4, label='hole real', vmin = 0, vmax = 0.6)
            fig.suptitle("N = " + str(N))
            fig.savefig(base_dir + '/'+ log +'/hole_matrix_'+str(N)+'.png')



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

    plt.close('all')
    fig, ax1 = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    ax1.plot(np.array(Ns),np.array(alpha_caps))
    ax1.set(xlabel='Number of trajectories', ylabel='Alpha')
    fig.suptitle("Alpha vs N")
    fig.savefig(base_dir + '/'+ log +'/alpha_curve.png')

    plt.close('all')
    fig, ax1 = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    ax1.plot(np.array(Ns),np.array(losses))
    ax1.set(xlabel='Number of trajectories', ylabel='Error')
    fig.suptitle("reward Error vs N")
    fig.savefig(base_dir + '/'+ log +'/Error_curve.png')

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