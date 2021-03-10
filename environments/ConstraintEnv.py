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

class ConstraintFrozenLake(gym.Wrapper):
    def __init__(self, env):
        super(ConstraintFrozenLake, self).__init__(env)
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


    def reset(self):
        self.step_ctr = 0
        self.reached = False
        return self.env.reset()


    def step(self, action):
        self.step_ctr += 1

        # Goal state is a absorbing state, increase score by 1
        if self.env.unwrapped.s == self.goal_state:
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

            self.s, r, done, info = self.env.step(applied_action)

            # If you reach a new constrain state, decrease score by 1
            if self.s in self.constraint_states:
                r = -self.constraint_costs[self.s]

            # If you reach a goal state, increase score by 1
            elif self.s == self.goal_state:
                self.reached = True
                r = 1
            
            # If you reach any other state, keep score the same
            else:
                r = 0

        return self.env.unwrapped.s, r, False, {"reached":self.reached, "counter":self.step_ctr}