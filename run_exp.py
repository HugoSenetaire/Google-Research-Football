import sys
import numpy as np
import argparse
import utils
from train import *
from agentLocal import DFPAgent, RandomAgent
from default_parameters import *
from kaggle_environments import make
import json
import pickle


if __name__ == '__main__':
    args = create_args()
    update_default_args(args)
    if args["load_model"]:
        print("Load parameter from {}".format(os.path.join(args["TOTAL_PATH"],'argument.json')))
        model_t = args["t"]
        with open(os.path.join(args["TOTAL_PATH"],'argument.pkl'), 'rb') as handle:
            args.update(pickle.load(handle))
            args["t"] = model_t
            args["load_model"] = True
    else :
        with open(os.path.join(args["TOTAL_PATH"],'argument.json'), 'w') as fp:
            json.dump(args, fp)
        with open(os.path.join(args["TOTAL_PATH"],'argument.pkl'), 'wb') as handle:
            pickle.dump(args, handle, protocol=pickle.HIGHEST_PROTOCOL)
    use_cuda = (torch.cuda.is_available() and args["gpu"])



    dfp_agent = DFPAgent(args, use_cuda=use_cuda)
    oppositionAgent = RandomAgent()
    
    optimizer = utils.create_optimizer(dfp_agent, args)
    if args["scheduler"]:
        scheduler = utils.create_scheduler(optimizer, args)
    else : 
        scheduler = None

    if args["load_model"]:
        if args["t"]==0 :
            utils.find_last_iteration(args)
        else :
            if not utils.check_weights(args):
                raise NameError("Iteration wanted do not exist")
        print("Load iteration {}".format(args["t"]))
        utils.load_model(dfp_agent, optimizer, scheduler, args)

    env = make("football", configuration={"save_video": args["SAVE_VIDEO"], "scenario_name": args["SCENARIO"], "running_in_notebook": args["running_in_notebook"], "episodeSteps": args["env_num_steps"]}, debug=args["DEBUG"])
    train(dfp_agent, env, optimizer, scheduler, args, list_opposition = [oppositionAgent])
