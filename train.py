import agentLocal
import model
import utils
import numpy as np




def train(dfp_agent, env, optimizer, args, list_opposition):
  agent = list_opposition[0]
  observation = env.reset()
  done=False
  epsilon = dfp_agent.initial_epsilon
  GAME = 0
  t = 0
  max_score = 0 # Maximum episode goal (Proxy for agent performance)
  score = 0
  # Buffer to compute rolling statistics 
  score_buffer = []
  r_t = 0
  goal = utils.create_goal([10,0.2,0.1,2], len(args["TIMESTEPS"]))
  loss = 0
  loss_queue_size = 50
  loss_queue = []
  loss_list = []
  list_iter = []
    

  ## Training loop
  while t<dfp_agent.explore + dfp_agent.observe:
    # print("joueur1")
    # print(observation[0])
    # print("joueur2")
    # print(observation[1])

    action_op = agent.get_action(observation[1]['observation'])

    frame_data = observation[0]["observation"]["players_raw"][0]
    sensory = utils.frame_data_to_tensor(frame_data, args["CHANNEL_NAMES"], args["IMAGE_SIZE"])
    measurements = utils.frame_data_to_measurements(observation[0], args["MEASUREMENT_NAMES"])

    if args["use_cuda"]:
        sensory, measurements, goal = sensory.cuda(), measurements.cuda(), goal.cuda()
      

    action_dfp = dfp_agent.get_action(sensory, measurements, goal, goal)
    try:
      observation= env.step([[action_dfp],action_op])
    except Exception as e:
      print("pre-mortem obs",observation)
      print(e)
      done=True

    # print("joueur1")
    # print(observation[0])
    # print("joueur2")
    # print(observation[1])

    ## TODO: Add frame skip between each memory ?
    is_terminated = observation[0]['status'] == "DONE"
    # break
    r_t = observation[0]['reward']
    score=r_t

    if (is_terminated):
        if (score > max_score):
            max_score = score
        GAME += 1
        score_buffer.append(score)
        print ("Episode Finish ")
        observation = env.reset()
    
    # save the sample <s, a, r, s'> to the replay memory and decrease epsilon
    dfp_agent.replay_memory(t, sensory, action_dfp, r_t, None, measurements, is_terminated)

    # Do the training
    if t > dfp_agent.observe and t % dfp_agent.timestep_per_train == 0:
        loss = dfp_agent.train_minibatch_replay(goal, optimizer).cpu().item()
        if len(loss_queue)==loss_queue_size :
          loss_queue.pop(0)
        loss_queue.append(loss)
        
        
    t += 1

    # save progress every 10000 iterations
    if t>dfp_agent.observe and t % 10000 == 0:
        print("Now we save model")
        torch.save(dfp_agent.model.state_dict(), "/content/drive/My Drive/google-football/weights.pth")

    # print info
    state = ""
    if t <= dfp_agent.observe:
        state = "observe"
    elif t > dfp_agent.observe and t <= dfp_agent.observe + dfp_agent.explore:
        state = "explore"
    else:
        state = "train"
    if (is_terminated):

        mean_loss = np.mean(loss_queue)
        
        
        print("TIME", t, "/ GAME", GAME, "/ STATE", state, \
              "/ EPSILON", dfp_agent.epsilon, "/ ACTION", action_dfp, "/ REWARD", r_t, \
              "/ goal", max_score, "/ LOSS", mean_loss)
        

      
        if t>dfp_agent.observe and len(loss_queue)>= loss_queue_size :
          list_iter.append(t)
          loss_list.append(mean_loss)
          with open('/content/drive/My Drive/google-football/metrics.csv','w') as f:
            f.write(str(list_iter).strip('[').strip(']') + "\n")
            f.write(str(loss_list).strip('[').strip(']'))