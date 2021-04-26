import gym
import os
import numpy as np

from environments.MyFrozenLakeEnv import actions_dict 
from utils.utility import create_trajs, is_invertible


def run():
    L = 16
    M = 4
    N = 100000
    traj_len = 30

    base_dir = "data/saved_trajectories8_noise" + str(N)
    
    # os.makedirs(base_dir,exist_ok=True)
    # save_loc = base_dir + "/" + str(traj_len)
    # os.makedirs(save_loc,exist_ok=True)
    trajs, R, C, K, H, alphas = create_trajs(N, L, M, traj_len)
    # H_alpha = np.multiply(np.tile(alphas, (64,1)).transpose(), H)
    # KH = np.concatenate((K,-H_alpha),axis=1) 
    # sym_mat = np.dot(KH.transpose(), KH)
    sym_mat = np.dot(K.transpose(), K)
    print("Calculated KtK")
    if is_invertible(sym_mat):
        pseudo_inv = np.dot(np.linalg.inv(sym_mat), K.transpose())
        r = np.dot(pseudo_inv, R)
        c = np.dot(pseudo_inv, C)

        print("rewards are:")
        for i in range(16):
            print(actions_dict[np.argmax(r[i*4:i*4+4])], end =" ")
            if (i+1)%4 == 0:
                print("\n")

        print("costs are:")
        for i in range(16):
            print(actions_dict[np.argmin(c[i*4:i*4+4])], end =" ")
            if (i+1)%4 == 0:
                print("\n")
    else:
        print("non-invertable")
    print("Done")




if __name__ == '__main__':
    run()