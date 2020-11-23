import agentLocal
import model
import utils
import numpy as np
import os
from tqdm import tqdm

def env_step(env, agent, observation, action_op, goal, channel_names, image_size, measurement_names, epsilon=None, use_cuda=False):

    frame_data = observation[0]["observation"]["players_raw"][0]
    sensory = utils.frame_data_to_tensor(frame_data, channel_names, image_size)
    measurements = utils.frame_data_to_measurements(observation[0], measurement_names)

    if use_cuda:
        sensory, measurements, goal = sensory.cuda(), measurements.cuda(), goal.cuda()
      
    action_dfp = agent.get_action(sensory, measurements, goal, goal, epsilon)
    observation= env.step([[action_dfp],action_op])

    return observation, action_dfp, sensory, measurements



def fill_replay_memory(dfp_agent,agent_opposition, env, observation, args, goal, score_buffer, GAME, max_score, num_step = 1):
  
  for i in range(num_step):
    action_op = agent_opposition.get_action(observation[1]['observation'])

    observation, action_dfp, sensory, measurements = env_step(env, dfp_agent, observation, action_op, goal, args['CHANNEL_NAMES'], args['IMAGE_SIZE'], args['MEASUREMENT_NAMES'], use_cuda=args['use_cuda'])

    ## TODO: Add frame skip between each memory ?
    is_terminated = observation[0]['status'] == "DONE"
    r_t = observation[0]['reward']
    score=r_t

    if (is_terminated):
        if (score > max_score):
            max_score = score
        GAME += 1
        score_buffer.append(score)
        print ("Episode Finished")
        observation = env.reset()
        if args['RANDOM_TRAIN_GOAL']:
          goal = utils.create_goal(list(np.random.rand(len( args['MEASUREMENT_NAMES']))), args["timesteps_goal"])
    
    # save the sample <s, a, r, s'> to the replay memory and decrease epsilon
    dfp_agent.replay_memory(sensory, action_dfp, r_t, None, measurements, is_terminated) # T est demandé mais pourquoi utiliser t ici ? Pas d'utilisation dans replay memory ? Besoin pour multithreading ?
    

  return score_buffer, GAME, max_score


def evaluation(eval_env, dfp_agent,agent, eval_goal, args):
    eval_rewards = []
    episode_lengths = []
    for i in tqdm(range(dfp_agent.nb_evaluation_episodes)):
      eval_observation = eval_env.reset()
      eval_episode_is_terminated = False
      episode_length=0
      while not eval_episode_is_terminated:
        action_op = agent.get_action(observation[1]['observation'])
        eval_observation, action_dfp, sensory, measurements = env_step(eval_env, dfp_agent, eval_observation, action_op, eval_goal, args['CHANNEL_NAMES'], args['IMAGE_SIZE'], args['MEASUREMENT_NAMES'], epsilon=0, use_cuda=args['use_cuda'])
        eval_episode_is_terminated = eval_observation[0]['status'] == "DONE"
        episode_length+=1
      eval_rewards.append(eval_observation[0]['reward'])
      episode_lengths.append(episode_length)
      #TODO: save logs of evaluation
    print("eval_rewards",eval_rewards)
    print("episode length", episode_lengths)
  


def train(dfp_agent, env, eval_env, optimizer, scheduler, args, list_opposition):
  agent = list_opposition[0]
  observation = env.reset()
  epsilon = dfp_agent.initial_epsilon
  GAME = 0
  t_train = args["t"]
  max_score = 0 # Maximum episode goal (Proxy for agent performance)
  score = 0

  # Buffer to compute rolling statistics 
  score_buffer = []
  r_t = 0
  if args['RANDOM_TRAIN_GOAL']:
    goal = utils.create_goal(list(np.random.rand(len(MEASUREMENT_NAMES))), args["timesteps_goal"])
  else:
    goal = utils.create_goal(args['EVAL_GOAL'], args["timesteps_goal"])
  eval_goal = utils.create_goal(args['EVAL_GOAL'], args["timesteps_goal"])
  loss = 0
  loss_queue_size = 50
  loss_queue = []
  loss_list = []
  list_iter = []
  

  ## Fill in the memory :
  print(f"Start observation for { dfp_agent.observe} steps")
  score_buffer, GAME, max_score = fill_replay_memory(dfp_agent, agent, env, observation, args, goal, score_buffer, GAME, max_score, num_step = dfp_agent.observe)
  print(f"End observation, start train")
  ## Training loop
  while t_train<args["total_train"]:
    
    
    score_buffer, GAME, max_score = fill_replay_memory(dfp_agent, agent, env, observation,  args, goal, score_buffer, GAME, max_score, num_step = dfp_agent.timestep_per_train)
    # Do the training
    loss = dfp_agent.train_minibatch_replay(goal, optimizer, scheduler).cpu().item()
    if len(loss_queue)==loss_queue_size :
      loss_queue.pop(0)
    loss_queue.append(loss)
 
    t_train += 1


    # save progress every args["save_every"] iterations (and make sure we don't save at first iteration)
    if t_train % args["save_every"] == 0:
        path = os.path.join(args["TOTAL_PATH"],"weights_{}.pth".format(t))
        print(f"Model saved with iteration of training {t_train} at path {path}")
        utils.save_model(t_train, optimizer, scheduler, dfp_agent, path)
    
    # Evaluate:
    if t_train%dfp_agent.evaluate_freq==0:
      evaluation(eval_env, dfp_agent,agent, eval_goal, args)


    # print info
    if t_train*dfp_agent.timestep_per_train <= dfp_agent.explore:
        state = "explore"
    else:
        state = "train"


    mean_loss = np.mean(loss_queue)
    print("STEP TOTAL",t_train * dfp_agent.timestep_per_train + dfp_agent.observe , "TIME TRAINED", t_train, "/ GAME", GAME, "/ STATE", state, \
          "/ EPSILON", dfp_agent.epsilon, "/ ACTION", action_dfp, "/ REWARD", r_t, \
          "/ goal", max_score, "/ LOSS", mean_loss)
    

  
    if len(loss_queue) >= loss_queue_size :
      list_iter.append(t)
      loss_list.append(mean_loss)
      with open(os.path.join(args['TOTAL_PATH'],"metrics.csv"),'w') as f:
        f.write(str(list_iter).strip('[').strip(']') + "\n")
        f.write(str(loss_list).strip('[').strip(']'))
