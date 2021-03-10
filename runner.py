import gym
import os
import numpy as np

from environments.ConstraintEnv import ConstraintFrozenLake
from utils.utility import create_trajs, is_invertible


def run():
    L = 16
    M = 4
    N = 100000
    traj_len = 30

    base_dir = "data/saved_trajectories8_noise" + str(N)
    
    os.makedirs(base_dir,exist_ok=True)
    save_loc = base_dir + "/" + str(traj_len)
    os.makedirs(save_loc,exist_ok=True)
    trajs, R, K, H = create_trajs(N, L, M, traj_len)
    KH = np.concatenate((K,H),axis=1) 
    sym_mat = np.dot(KH.transpose(), KH)
    print("Calculated KtK")
    if is_invertible(sym_mat):
        pseudo_inv = np.dot(np.linalg.inv(sym_mat), K.transpose())
    else:
        print("non-invertable")
    print("Done")




if __name__ == '__main__':
    run()