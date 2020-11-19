import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import optim
import glob
import os

TOP_LEFT_CORNER_COORDS = (-1.1, -0.45)



def coords_to_pixel(x,y,image_size):
  """
  Transforms float coordinates to associate pixel coordinates in image of image_size
  """
  v = int((1 -x/TOP_LEFT_CORNER_COORDS[0]) / 2 * (image_size[1]-1))
  u = int((1 -y/TOP_LEFT_CORNER_COORDS[1]) / 2 * (image_size[0]-1))
  return u,v

def frame_data_to_channel(frame_data, channel_name, channel_shape):
  """
  Mapping for each type of data to the associated 2D channel
  """
  channel = torch.zeros(channel_shape, dtype=torch.float)
  if channel_name in ['left_team', 'right_team']:
    for coords in frame_data[channel_name]:
      channel[coords_to_pixel(*coords, channel_shape)]=1
  elif channel_name=="ball_direction":
    #TODO: add height of ball to data
    channel[coords_to_pixel(frame_data[channel_name][0] + frame_data['ball'][0],frame_data[channel_name][1] + frame_data['ball'][1], channel_shape)]=1
  elif channel_name == "ball":
    channel[coords_to_pixel(*frame_data[channel_name][:2], channel_shape)]=1
  elif 'direction' in channel_name:
    aux_channel_name = channel_name.split('_direction')[0]
    for i in range(len(frame_data[channel_name])):
      channel[coords_to_pixel(frame_data[channel_name][i][0] + frame_data[aux_channel_name][i][0], frame_data[channel_name][i][1] + frame_data[aux_channel_name][i][1], channel_shape)]=1
  return channel

def frame_data_to_tensor(frame_data, channel_names, channel_shape):
  """
  Build tensor from the frame_data extracted from each channel in channel_names
  """
  tensor = torch.zeros((len(channel_names),) + channel_shape, dtype=torch.float )
  for i,channel_name in enumerate(channel_names):
    tensor[i,:,:]= frame_data_to_channel(frame_data, channel_name, channel_shape)
  return tensor

def frame_data_to_measurement(frame_data, measurement_name):
  if measurement_name=="ball_distance_to_center": #distance au centre (latéralement)
    return 1 - np.abs(frame_data['observation']['players_raw'][0]["ball"][1])/0.42
  if measurement_name=="ball_distance_to_goal":
    return (1 + frame_data['observation']['players_raw'][0]["ball"][0])/2
  if measurement_name=="goals": #this is the goal differential, maybe separate team goal from opponents
    return frame_data['reward']
  if measurement_name=="possession":
    ## frame_data['ball_owned_team'] is such as -1 = ball not owned, 0 = left team, 1 = right team.
    ## TODO: Maybe redesign the ball owned measurement when no team has control of the ball
    ## TODO: transformer en derniere equipe a avoir toucher la balle ?
    mapping = {
        0: -1,
        -1:0,
        1:1
    }
    return mapping[frame_data['observation']['players_raw'][0]['ball_owned_team']]

def frame_data_to_measurements(frame_data,measurement_names):
  """
  Attention c'est pas les memes frame_data qui sont attendues ici que pour construire le tensor (sensory stream), a changer (au moins les noms)
  """
  #TODO: doit retourner un tensor de taille (len(measurement_names),). le type est à déterminer
  measurements = []
  for measurement_name in measurement_names:
    measurements.append(frame_data_to_measurement(frame_data, measurement_name))
  return torch.tensor(measurements, dtype=torch.float)

def raw_data_to_target(episode_data, start_frame, measurement_names, future_timesteps):
  """
  future_timesteps must be sorted in increasing order
  """
  done = False
  future_measurements = torch.zeros((len(future_timesteps), len(measurement_names)))
  last_offset = 0
  timesteps_count = 0
  current_measurements = frame_data_to_measurements(episode_data[start_frame][0], measurement_names)
  for j in range(1, future_timesteps[-1]+1):
    if start_frame+j >= len(episode_data):
      done = True
      if j in future_timesteps:
        future_measurements[timesteps_count] = frame_data_to_measurements(episode_data[start_frame+last_offset][0], measurement_names) - current_measurements
        timesteps_count+=1
    else: #if episode is not finished
      if j in future_timesteps:
        if not done:
          future_measurements[timesteps_count] = frame_data_to_measurements(episode_data[start_frame+j][0], measurement_names) - current_measurements
          timesteps_count+=1
          last_offset = j
        else:
          future_measurements[timesteps_count] = frame_data_to_measurements(episode_data[start_frame+last_offset][0], measurement_names) - current_measurements
          timesteps_count+=1
  return future_measurements



def create_optimizer(dfp_agent,args):
  if args["optimizer"] == "Adam":
    optimizer = optim.Adam(dfp_agent.model.parameters(), lr = args["lr"])
  elif args["optimizer"] == "SGD":
    optimizer = optim.SGD(dfp_agent.model.parameters(), lr=args["lr"], momentum=args["momentum"])
  else :
    raise NotImplementedError
  return optimizer


def create_scheduler(optimizer,args):
  if args["scheduler_type"] == "exponential":
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer,args["scheduler_rate"])
  else :
    raise NotImplementedError
  return scheduler


def create_goal(rel_goal, timesteps_size):
  """
  Create goal with given relative importance of measurements for the LAST timestep to predict (like in the paper)
  """
  goal = torch.zeros((timesteps_size, len(rel_goal)), dtype=torch.float)
  goal[-1]=torch.tensor(rel_goal)
  goal = goal.flatten()
  return goal


def save_model(t, optimizer, scheduler, dfp_agent, path):
  if scheduler is None :
    torch.save({
        't': t,
        'model_state_dict': dfp_agent.model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
  else :
      torch.save({
        't': t,
        'model_state_dict': dfp_agent.model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, path)


def check_weights(args):
  total_path = args["TOTAL_PATH"]
  if not os.path.exists(os.path.join(total_path, "weights_{}.pth".format(args["t"]))):
    return False
  else : 
    return True
  
def find_last_iteration(args):
  total_path = args["TOTAL_PATH"]
  list_trained = glob.glob(os.path.join(total_path,"*.pth"))
  if not os.path.exists or len(list_trained) == 0 :
    raise NameError("No experiments previously trained at this path")
  list_t = list(map(lambda x: int(x.strip(os.path.join(total_path, "weights_")).strip(".pth")),list_trained)) # TODO : Changer ce monstre
  list_t.sort()
  args["t"] = list_t[-1]



def load_model(dfp_agent, optimizer, scheduler, args):
  t = args["t"]
  path = os.path.join(args["TOTAL_PATH"],"weights_{}.pth".format(t))
  checkpoint = torch.load(path)
  dfp_agent.model.load_state_dict(checkpoint['model_state_dict'])
  dfp_agent.epsilon -= t * (dfp_agent.initial_epsilon - dfp_agent.final_epsilon) / dfp_agent.explore
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  if scheduler is not None :
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

