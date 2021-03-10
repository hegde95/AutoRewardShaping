import gym
import numpy as np
import random

base_dir = "data/saved_trajectories8_noise"
# # base_dir = "saved_trajectories8_noise50000"

def get_Reward_mat(T = 14):
    return np.loadtxt(base_dir + "/30/reward_mat.csv", delimiter=' ')


            

actions_dict = {
    0:"Left",
    1:"Down",
    2:"Right",
    3:"Up",
    # 4:"Noop"
}

class AbsorbingFrozenLake(gym.Wrapper):
    def __init__(self, env, reward_shaped = False, probabilistic = True, max_len = 30):
        super(AbsorbingFrozenLake, self).__init__(env)
        self.step_ctr = 0
        self.max_length = max_len
        self.probabilistic = probabilistic
        self.list_of_actions = [0,1,2,3]
        self.reached = False
        self.reward_shaped = reward_shaped
        if self.reward_shaped:
            self.R_mat = get_Reward_mat()
        
    def step(self, action):
        self.step_ctr += 1

        # Consider the absorbing states, if already in these states, then actions have no consiquence, decrease score by 1
        if self.env.unwrapped.s in (19, 29, 35, 41, 42, 46, 49, 52, 54, 59):
            if self.reward_shaped:
                r = self.R_mat[self.env.unwrapped.s, action]
            else:
                r = -1
            return self.env.unwrapped.s, r, True, {"reached":self.reached, "counter":self.step_ctr, "prob":None}


        # Goal state is also a absorbing state, increase score by 1
        elif self.env.unwrapped.s == 63:
            if self.reward_shaped:
                r = self.R_mat[self.env.unwrapped.s, action]
            else:
                r = 1
            return self.env.unwrapped.s, r, True, {"reached":self.reached, "counter":self.step_ctr, "prob":None}
        
        
        # If not in an absorbing state apply the action on the unwrapped env
        else:
            if self.probabilistic:
                ran = np.random.rand()
                if ran < 0.8:
                    applied_action = action
                else:
                    applied_action = random.choice([ac for ac in self.list_of_actions if ac!= action])
            else:
                applied_action = action

            self.s, r, done, info = self.env.step(applied_action)

            # If you reach a new absorbing state, decrease score by 1
            if self.s in (19, 29, 35, 41, 42, 46, 49, 52, 54, 59):
                if self.reward_shaped:
                    r = self.R_mat[self.env.unwrapped.s, action]
                else:
                    r = -1
                return self.env.unwrapped.s, r, True, {"reached":self.reached, "counter":self.step_ctr, "prob":info["prob"]}
            
            # If you reach a goal state, increase score by 1
            elif self.s == 63:
                self.reached = True
                if self.reward_shaped:
                    r = self.R_mat[self.env.unwrapped.s, action]
                else:
                    r = 1
                return self.env.unwrapped.s, r, True, {"reached":self.reached, "counter":self.step_ctr, "prob":info["prob"]}
            
            # If you reach any other state, keep score the same
            else:
                if self.reward_shaped:
                    r = self.R_mat[self.env.unwrapped.s, action]
                else:
                    r = 0
                return self.env.unwrapped.s, r, False, {"reached":self.reached, "counter":self.step_ctr, "prob":info["prob"]}

    def reset(self):
        self.step_ctr = 0
        self.reached = False
        return self.env.reset()