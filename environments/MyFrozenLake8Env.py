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

class MyFrozenLake8(gym.Env):
    def __init__(self, probabilistic = True, random_reset = True, absorbing_goal_state = False):
        self.list_of_actions = [0, 1, 2, 3]
        self.probabilistic = probabilistic
        self.random_reset = random_reset
        self.absorbing_goal_state = absorbing_goal_state
        self.probability_of_choosing_right_action = 0.8

        self.dim = 8

        # self.constraint_states = [33, 42, 51, 49, 59, 56, 7, 14, 22, 23, 31, 20, 4, 12, 11, 3]
        self.constraint_states = [35, 20]
        # self.constraint_states = [35]

        # self.state_rewards = {
        #     0:0.1,    1:0.2,    2:0.3,    3:0.2,    4:0.2,    5:0.1,     6:0.4,    7:0.3,   
        #     8:0.1,    9:0.2,    10:0.4,   11:0.2,   12:0.1,   13:0.2,   14:0.3,   15:0.2,
        #     16:0.1,   17:0.2,   18:0.3,   19:0.2,   20:0.2,   21:0.1,   22:0.4,   23:0.3,   
        #     24:0.1,   25:0.2,   26:0.4,   27:0.2,   28:0.1,   29:0.2,   30:0.3,   31:0.2,
        #     32:0.1,   33:0.2,   34:0.3,   35:0.2,   36:0.2,   37:0.1,   38:0.4,   39:0.3,   
        #     40:0.1,   41:0.2,   42:0.4,   43:0.2,   44:0.1,   45:0.2,   46:0.3,   47:0.2,
        #     48:0.1,   49:0.2,   50:0.3,   51:0.2,   52:0.2,   53:0.1,   54:0.4,   55:0.3,   
        #     56:0.1,   57:0.2,   58:0.4,   59:0.2,   60:0.1,   61:0.2,   62:0.3,   63:0.2,
        #     }
        self.state_rewards = {
            0:0.0,    1:0.0,    2:0.0,    3:0.0,    4:0.0,    5:0.0,     6:0.0,    7:0.0,   
            8:0.0,    9:0.1,    10:0.1,   11:0.1,   12:0.1,   13:0.1,   14:0.1,   15:0.1,
            16:0.0,   17:0.1,   18:0.2,   19:0.2,   20:0.2,   21:0.2,   22:0.2,   23:0.2,   
            24:0.0,   25:0.1,   26:0.2,   27:0.3,   28:0.3,   29:0.3,   30:0.3,   31:0.3,
            32:0.0,   33:0.1,   34:0.2,   35:0.3,   36:0.4,   37:0.4,   38:0.4,   39:0.4,   
            40:0.0,   41:0.1,   42:0.2,   43:0.3,   44:0.4,   45:0.5,   46:0.5,   47:0.5,
            48:0.0,   49:0.1,   50:0.2,   51:0.3,   52:0.4,   53:0.5,   54:0.6,   55:0.6,   
            56:0.0,   57:0.1,   58:0.2,   59:0.3,   60:0.4,   61:0.5,   62:0.6,   63:0.7,
            }
        self.goal_state = (self.dim * self.dim) - 1
        self.top_wall = [ n for n in range(self.dim)]
        self.bottom_wall = [ 63 - n for n in range(self.dim)]
        self.left_wall = [ self.dim * n for n in range(self.dim)]
        self.right_wall = [ self.dim * (n +1) - 1 for n in range(self.dim)]

        self.reached = False
        self.step_ctr = 0

        if self.absorbing_goal_state:
            self.state_rewards[self.goal_state] = 1

        # states are:
        # 0    1    2    3    4    5     6    7   
        # 8    9    10   11   12   13   14   15
        # 16   17   18   19   20   21   22   23   
        # 24   25   26   27   28   29   30   31
        # 32   33   34   35   36   37   38   39   
        # 40   41   42   43   44   45   46   47
        # 48   49   50   51   52   53   54   55   
        # 56   57   58   59   60   61   62   63

    def set_alpha(self, alpha):
        self.alpha = alpha

    def get_dist(self, state1):
        x1 = self.current_state % self.dim
        y1 = self.current_state // self.dim

        x2 = state1 % self.dim
        y2 = state1 // self.dim

        return abs(x1-x2) + abs(y1-y2)

    def get_min_hole_dist(self):
        min_dist = 9999
        for hole in self.constraint_states:
            dist_to_hole = self.get_dist(hole)
            min_dist = min(min_dist, dist_to_hole)
        
        return min_dist


    def reset(self):
        self.step_ctr = 0
        self.reached = False
        self.alpha = 1
        if self.random_reset:
            self.current_state = np.random.randint(0,63)
        else:
            self.current_state = 0
        return self.current_state

    def apply_action(self, action):
        if (self.current_state in self.top_wall and action == 3) \
            or (self.current_state in self.bottom_wall and action == 1)\
            or (self.current_state in self.left_wall and action == 0)\
            or (self.current_state in self.right_wall and action == 2):
            pass
        else:
            if action == 0:
                self.current_state += -1
            elif action == 1:
                self.current_state += self.dim
            elif action == 2:
                self.current_state += 1
            elif action == 3:
                self.current_state += -self.dim

    def step(self, action):
        self.step_ctr += 1

        # Goal state is a absorbing state, increase score by 1
        if (self.current_state == self.goal_state) and self.absorbing_goal_state:
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

            # # If you reach a new constrain state, decrease score by cost times alpha
            # if self.current_state in self.constraint_states:
            #     # r = -self.alpha * self.constraint_costs[self.current_state]
            #     r = 0

            # # If you reach a goal state, increase score by 1
            # elif self.current_state == self.goal_state:
            #     self.reached = True
            #     r = 1
            
            # # If you reach any other state, keep score the same
            # else:
            #     r = self.state_rewards[self.current_state]
            #     # r = 0
            r = self.state_rewards[self.current_state]

        return self.current_state, r, False, self.get_info()

    def get_info(self):
        return {"reached":self.reached, "counter":self.step_ctr, "min_hole_dist":self.get_min_hole_dist()}