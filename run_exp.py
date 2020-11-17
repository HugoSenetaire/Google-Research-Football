import sys
import numpy as np
import argparse
import utils
import train
from agentLocal import DFPAgent, RandomAgent
from default_parameters import *

if __name__ == '__main__':
    args = create_args()
    update_default_args(args)
    use_cuda = (torch.cuda.is_available() and args["gpu"])
    print(args)
    dfp_agent = DFPAgent((len(args["CHANNEL_NAMES"]),)+args["IMAGE_SIZE"], args["MEASUREMENT_NAMES"], args["NB_ACTIONS"], args["TIMESTEPS"], use_cuda=use_cuda)
    oppositionAgent = RandomAgent()
    optimizer = utils.create_optimizer(dfp_agent, args)
    env = make("football", configuration={"save_video": True, "scenario_name": "11_vs_11_kaggle", "running_in_notebook": False, "episodeSteps": args["NUM_STEPS"]}, debug=True)
    train(dfp_agent, env, optimizer, args, list_opposition = [oppositionAgent])