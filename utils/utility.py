import numpy as np
from tqdm import tqdm
import gym

from environments.ConstraintEnv import ConstraintFrozenLake
from environments.MyFrozenLakeEnv import MyFrozenLake

def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]

def create_trajs(N, L, M, maxlen, Noise = True):
    fin = {}
    R = []
    C = []
    alphas = []
    
    # State-Action visitation frequency
    K = np.zeros((N, L*M))

    # Next - State visitation frequency
    H = np.zeros((N, L*M))

    # create environment
    # env = gym.make('FrozenLake-v0', is_slippery=False, map_name="4x4")
    # env = ConstraintFrozenLake(env)
    env = MyFrozenLake(probabilistic=False)

    # Nested function def to run a given trajectory and update the occupancy matrices
    def run_traj(traj, l, maxlen, alpha = 1):
        # reset everything
        st = env.reset()
        
        env.set_alpha(alpha)
        done = False
        k = 0
        score = 0
        total_cost = 0

        # run through the pre determined trajectory
        while k < len(traj):
            action = traj[k]
            k += 1
            ij_index = st * M + int(action)
            
            next_st, reward, done, _ = env.step(action)

            # Increment occupation measure as we need the state action pair before the action is taken to predict reward
            K[l,ij_index] += 1



            # H matrix section
            ############################################################################################################################################
            # # If next state is in the constriant state, icrease occupancy measure of H for the original state action pair
            if next_st in env.constraint_states:
                H[l,ij_index] += 1
                total_cost += reward
            
            # # Here we shall try to increase the occupancy measure of the next state only
            # if next_st in env.constraint_states:
            #     H[l,next_st] += 1

            # Here try making the cost a function of state, action and next state
            # if next_st in env.constraint_states:
            #     iji_dash_index = st * M + int(action) * L + int(next_st)
            #     H[l,next_st] += 1
            ############################################################################################################################################
            else:
                score += reward

            st = next_st
        return score, total_cost


    for j in tqdm(range(N)):
        # Step1 - Create a random trajectory
        traj = list(np.random.choice([0,1,2,3], maxlen, p = [0.1,0.4,0.4,0.1]))

        # Step2 - Run the trajectory through the environment and update the Occupency matrices
        # alpha = np.random.rand()
        alpha = 1
        alphas.append(alpha)
        score, total_cost = run_traj(traj, j, maxlen, alpha)
        R.append(score)
        C.append(total_cost)

        # This is just to create a map of list of trajectories by their scores
        if score in fin.keys():
            fin[score].append(traj)
        else:
            fin[score] = [traj]

    R = np.array(R)
    C = np.array(C)
    # Add gaussian noise to the score
    if Noise:
        R = abs(np.random.normal(R,1))
        C = -abs(-np.random.normal(C,0.5))

    return fin, R, C, K, H, np.array(alphas)
