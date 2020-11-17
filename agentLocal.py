from collections import deque
import random
import matplotlib.pyplot as plt
from model import *
#Taken from https://github.com/flyyufelix/Direct-Future-Prediction-Keras/blob/master/dfp.py


from kaggle_environments.envs.football.helpers import *

@human_readable_agent
class RandomAgent():
    def __init__(self):
        print("Random agent")


    def get_action(obs):
        # Make sure player is running.
        if Action.Sprint not in obs['sticky_actions']:
            return Action.Sprint
        # We always control left team (observations and actions
        # are mirrored appropriately by the environment).
        controlled_player_pos = obs['left_team'][obs['active']]
        # Does the player we control have the ball?
        if obs['ball_owned_player'] == obs['active'] and obs['ball_owned_team'] == 0:
            # Shot if we are 'close' to the goal (based on 'x' coordinate).
            if controlled_player_pos[0] > 0.5:
                return Action.Shot
            # Run towards the goal otherwise.
            return Action.Right
        else:
            # Run towards the ball.
            if obs['ball'][0] > controlled_player_pos[0] + 0.05:
                return Action.Right
            if obs['ball'][0] < controlled_player_pos[0] - 0.05:
                return Action.Left
            if obs['ball'][1] > controlled_player_pos[1] + 0.05:
                return Action.Bottom
            if obs['ball'][1] < controlled_player_pos[1] - 0.05:
                return Action.Top
            # Try to take over the ball if close to the ball.
            return Action.Slide


class DFPAgent():
    def __init__(self, state_size, measurement_names, action_size, timesteps, use_cuda=False):
      self.measurement_names = measurement_names
      # get size of state, measurement, action, and timestep
      self.state_size = state_size #sensory input size
      self.measurement_size = len(measurement_names)
      self.action_size = action_size
      self.timesteps = timesteps

      # these is hyper parameters for the DFP
      self.epsilon = 1.0
      self.initial_epsilon = 1.0
      self.final_epsilon = 0.0001
      self.batch_size = 32
      self.observe = 2000
      self.explore = 50000 
      self.frame_per_action = 4
      self.timestep_per_train = 5 # Number of timesteps between training interval

      # experience replay buffer
      self.memory = deque()
      self.max_memory = 20000

      # create model
      self.model = DFPBasicModel(self.state_size, self.measurement_size, len(self.timesteps), self.action_size)
      self.use_cuda = use_cuda
    
    def get_action(self, state, measurement, goal, inference_goal):
        """
        Get action from model using epsilon-greedy policy
        """
        if np.random.rand() <= self.epsilon:
            #print("----------Random Action----------")
            action_idx = random.randrange(self.action_size)
        else:
            state=state.unsqueeze(dim=0)
            measurement = measurement.unsqueeze(dim=0)
            goal = goal.unsqueeze(dim=0)
            f = self.model(state, measurement, goal)
            f_pred = f.squeeze(dim=0)
            obj = f_pred@inference_goal # num_action
            action_idx = torch.argmax(obj).cpu().item()
        return action_idx
    
    # Save trajectory sample <s,a,r,s'> to the replay memory
    def replay_memory(self, s_t, action_idx, r_t, s_t1, m_t, is_terminated):
        self.memory.append((s_t, action_idx, r_t, s_t1, m_t, is_terminated))
        if self.epsilon > self.final_epsilon and t > self.observe:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.explore

        if len(self.memory) > self.max_memory:
            self.memory.popleft()
    
    def get_target(self, start_frame):
        future_measurements = []
        last_offset = 0
        done = False
        for j in range(self.timesteps[-1]+1):
            if not self.memory[start_frame+j][5]: # if episode is not finished
                if j in self.timesteps: # 1,2,4,8,16,32
                    if not done:
                        future_measurements += list( (self.memory[start_frame+j][4] - self.memory[start_frame][4]) )
                        last_offset = j
                    else:
                        future_measurements += list( (self.memory[start_frame+last_offset][4] - self.memory[start_frame][4]) )
            else:
                done = True
                if j in self.timesteps: # 1,2,4,8,16,32
                    future_measurements += list( (self.memory[start_frame+last_offset][4] - self.memory[start_frame][4]) )
        return torch.tensor(future_measurements)
    
    # Pick samples randomly from replay memory (with batch_size)
    def train_minibatch_replay(self, goal, optimizer):
        """
        Train on a single minibatch
        """
        batch_size = min(self.batch_size, len(self.memory))
        rand_indices = np.random.choice(len(self.memory)-(self.timesteps[-1]+1), self.batch_size)

        state_input = torch.zeros(((batch_size,) + self.state_size)) # Shape batch_size, img_rows, img_cols, 4
        measurement_input = torch.zeros((batch_size, self.measurement_size))
        goal_input = goal.repeat((batch_size,1))
        f_action_target = torch.zeros((batch_size, (self.measurement_size * len(self.timesteps)))) 
        action = []

        for i, idx in enumerate(rand_indices):
            f_action_target[i,:] = self.get_target(idx)
            state_input[i,:,:,:] = self.memory[idx][0]
            measurement_input[i,:] = self.memory[idx][4]
            action.append(self.memory[idx][1])
        
        if self.use_cuda:
          state_input, measurement_input, goal_input = state_input.cuda(), measurement_input.cuda(), goal_input.cuda()
          
        f_target = self.model(state_input, measurement_input, goal_input)

        for i in range(self.batch_size):
            f_target[i, action[i]] = f_action_target[i]
          
        loss = train_on_batch(self.model, optimizer, state_input, measurement_input, goal_input, f_target)

        return loss
    
    # load the saved model
    def save_model(self, path = "/content/drive/My Drive/google-football/weights.pth"):
        torch.save(self.model.state_dict(), path)
        self.model.load_weights(name)

    #  save the model which is under training
    def load_model(self, path):
        self.model.load_state_dict(path)