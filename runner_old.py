from random import sample 
from Environment import MODFrozenLake, actions_dict, base_dir
import gym
import csv
import os
from tqdm import tqdm
import numpy as np

L = 64
M = 4
N = 10000

base_dir = base_dir + str(N)

Noise = True

def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]

def make_matrix(r):
    R = np.zeros((L,M))
    for k,reward in enumerate(r):
        i = k // M
        j = k % M
        R[i,j] = reward
    return R

def comb_main(traj_len = 9, s = 4):
    os.makedirs(base_dir,exist_ok=True)
    save_loc = base_dir + "/" + str(traj_len)
    os.makedirs(save_loc,exist_ok=True)
    trajs, R, K = create_trajs(traj_len, s)
    sym_mat = np.dot(K.transpose(), K)
    print("Calculated KtK")
    if is_invertible(sym_mat):
        pseudo_inv = np.dot(np.linalg.inv(sym_mat), K.transpose())
        print("Calculated pseudo inverse")
        r = np.dot(pseudo_inv, R)
        dir_path = os.path.join(base_dir, str(traj_len))
        np.savetxt(os.path.join(dir_path, "reward.csv"), r)
        r_mat = make_matrix(r)
        np.savetxt(os.path.join(dir_path, "reward_mat.csv"), r_mat)
        print("Saved r")
        for l in trajs.keys():
            file_path = save_loc + "/" + str(l) + ".csv"
            with open(file_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(trajs[l])
    else:
        print("non-invertable")


def create_trajs(len = 6, s = 4):
    fin = {}
    R = []
    K = np.zeros((N, L*M))

    for j in tqdm(range(N)):
        # traj = list(np.random.randint(low = 0,high=4,size=len))
        traj = list(np.random.choice([0,1,2,3], len, p = [0.1,0.4,0.4,0.1]))
        K, score = run_traj(K, traj, j, s, len)
        R.append(score)
        if score in fin.keys():
            fin[score].append(traj)
        else:
            fin[score] = [traj]
    R = np.array(R)
    if Noise:
        R = np.random.normal(R,2)
    return fin, R, K

def run_traj(K, traj, l, s = 4, maxlen = 30):
    if s == 4:
        map_name = "4x4"
    else:
        map_name = "8x8"
    env = gym.make('FrozenLake-v0', is_slippery=False, map_name=map_name)
    env = MODFrozenLake2(env, reward_shaped = False, probabilistic=True, max_len=maxlen)
    st = env.reset()
    done = False
    k = 0
    complete = False
    score = 0
    while k < len(traj):
        action = traj[k]
        k += 1
        ij_index = st * M + int(action)
        K[l,ij_index] += 1
        st, reward, done, _ = env.step(action)
        score += reward
    return K, score

if __name__ == '__main__':
    comb_main(60,8)