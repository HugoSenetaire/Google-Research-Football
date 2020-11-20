import argparse
import torch
import datetime
import os


def create_args():
    parser = argparse.ArgumentParser(description='DFP exp')
    parser.add_argument('--gpu', action='store_true', help='Whether to use GPU')
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--global_path', type=str)
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--t', type = int, default=0) # when -1, use the last weights saved
    parser.add_argument('--running_in_notebook', action = 'store_true')
    parser.add_argument('--env_num_steps', type=int, default=200)
    args = parser.parse_args()
    return vars(args)


def get_total_path(args):
    if args["global_path"] is None:
        if args["running_in_notebook"]:
            args["global_path"] =  "/content/drive/My Drive/google-football/"
        else:
            args["global_path"] = "experiments/"
    
    if args["experiment_name"] is None :
        currentdate = str(datetime.datetime.now())
        args["experiment_name"] = args["SCENARIO"] + "_" + currentdate  
        
    total_path = os.path.join(args["global_path"],args["experiment_name"])

    return total_path



def update_default_args(args):

    # Settings update :

    args.update({
        "DEBUG": False,
        "SAVE_VIDEO" : False,
    })



    # Games update :
    args.update({
        "SCENARIO" : "11_vs_11_kaggle",
        "CHANNEL_NAMES" : ["left_team", 'right_team', 'left_team_direction', 'right_team_direction', 'ball', 'ball_direction', "current_player"],
        "MEASUREMENT_NAMES" : ['goals', 'ball_distance_to_goal', "ball_distance_to_center", "possession"],
        "TIMESTEPS" : [1,2,4,8,16,32],
        "IMAGE_SIZE" : (43, 101),
        "NB_ACTIONS" : 19,
        "RANDOM_TRAIN_GOAL": False, #If false, train goal is equal to eval goal
        "EVAL_GOAL": [10,0.2,0.1,2]
    })

        # Save update

    args["save_every"] = 5000
        
    total_path = get_total_path(args)
    if not os.path.exists(total_path):
        os.makedirs(total_path)

    args["TOTAL_PATH"] = total_path


    # DFP Agent update :
    args.update({
        "lambda_relu" : 0.05,
        "gamma" : 0.99,
        "epsilon" : 1.0,
        "initial_epsilon" : 1.0,
        "final_epsilon": 0.0001,
        "batch_size" : 64,
        "explore" : 10000000,
        "observe" : 2000,
        "frame_per_action" : 1, # TODO : USELESS ?
        "timestep_per_train" : 64,
        "max_memory" : 20000,
        "agent_evaluate_freq": 5000,
        "nb_evaluation_episodes": 20 #is that too much / too little (depends on scenario)
    })


    args["total_train"] = 11000000
    assert(args["total_train"]>=args["explore"]+args["observe"])

    #Optimizer update :
    args.update({
        "scheduler" : True,
        "scheduler_rate" : 0.999,
        "scheduler_type" : "exponential",
        "lr" : 0.0005,
        "momentum" : 0.9,
        "seed" : 1,
        "optimizer" : "Adam", # 
        #"optimizer" = "SGD"
    })
    