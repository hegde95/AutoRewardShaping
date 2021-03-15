import gym
import numpy as np
import random



actions_dict = {
    0:"Left",
    1:"Down",
    2:"Right",
    3:"Up",
    # 4:"Noop"
}

class MyFrozenLake(gym.Env):
    def __init__(self):
        self.list_of_actions = [0, 1, 2, 3]
        self.probabilistic = True
        self.probability_of_choosing_right_action = 0.8

        self.constraint_states = [5, 7, 11, 12]
        self.constraint_costs = {5:0.4,
                                7:0.3,
                                11:0.2,
                                12:0.1}
        self.goal_state = 15

        self.reached = False
        self.step_ctr = 0

        # states are:
        # [ 0, 1, 2, 3]
        # [ 4, 5, 6, 7]
        # [ 8, 9,10,11]
        # [12,13,14,15]

    def set_alpha(self, alpha):
        self.alpha = alpha


    def reset(self):
        self.step_ctr = 0
        self.reached = False
        self.alpha = 1
        self.current_state = 0
        return self.current_state

    def apply_action(self, action):
        if (self.current_state in [ 0, 1, 2, 3] and action == 3) \
            or (self.current_state in [12,13,14,15] and action == 1)\
            or (self.current_state in [0,4,8,12] and action == 0)\
            or (self.current_state in [3,7,11,15] and action == 2):
            pass
        else:
            if action == 0:
                self.current_state += -1
            elif action == 1:
                self.current_state += 4
            elif action == 2:
                self.current_state += 1
            elif action == 3:
                self.current_state += -4

    def step(self, action):
        self.step_ctr += 1

        # Goal state is a absorbing state, increase score by 1
        if self.current_state == self.goal_state:
            r = 1
            # do nothing else as this is an absorbing state

        # If not in an absorbing state apply the action on the unwrapped env
        else:
            if self.probabilistic:
                ran = np.random.rand()
                if ran < self.probability_of_choosing_right_action:
                    applied_action = action
                else:
                    applied_action = random.choice([ac for ac in self.list_of_actions if ac!= action])
            else:
                applied_action = action

            # self.s, r, done, info = self.env.step(applied_action)
            self.apply_action(applied_action)

            # If you reach a new constrain state, decrease score by cost times alpha
            if self.current_state in self.constraint_states:
                r = -self.alpha * self.constraint_costs[self.current_state]

            # If you reach a goal state, increase score by 1
            elif self.current_state == self.goal_state:
                self.reached = True
                r = 1
            
            # If you reach any other state, keep score the same
            else:
                r = 0

        return self.current_state, r, False, {"reached":self.reached, "counter":self.step_ctr}