import argparse
import torch
import datetime
import os


def create_args():
    parser = argparse.ArgumentParser(description='DFP exp')
    parser.add_argument('--gpu', action='store_true', help='Whether to use GPU')
    parser.add_argument('--experiment_name', type=str, default = "None")
    parser.add_argument('--GLOBAL_PATH', type=str, default="experiments/")
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--t', type = int, default=0) # when -1, use the last weights saved
    args = parser.parse_args()
    return vars(args)



def update_default_args(args):

    # Settings update :

    args.update({
        "running_in_notebook" : True,
        "DEBUG": False,
        "SAVE_VIDEO" : False,
    })



    # Games update :
    args.update({
        "SCENARIO" : "11_vs_11_kaggle",
        "CHANNEL_NAMES" : ["left_team", 'right_team', 'left_team_direction', 'right_team_direction', 'ball', 'ball_direction'],
        "MEASUREMENT_NAMES" : ['goals', 'ball_distance_to_goal', "ball_distance_to_center", "possession"],
        "TIMESTEPS" : [1,2,4,8,16,32],
        "IMAGE_SIZE" : (43, 101),
        "NB_ACTIONS" : 19,
        "NUM_STEPS" : 200
    })

        # Save update

    args["save_every"] = 5000

    if args["GLOBAL_PATH"] == "experiments/":
        if args["running_in_notebook"]:
            args["GLOBAL_PATH"] =  "/content/drive/My Drive/google-football/"
    
    if args["experiment_name"]=="None" :
        currentdate = str(datetime.datetime.now())
        args["experiment_name"] = args["SCENARIO"] + "_" + currentdate  
        
    total_path = os.path.join(args["GLOBAL_PATH"],args["experiment_name"])
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
        "batch_size" : 32,
        "explore" : 100000,
        "observe" : 2000,
        "frame_per_action" : 4,
        "timestep_per_train" : 5,
        "max_memory" : 20000,

    })


    args["total_train"] = 2000000
    assert(args["total_train"]>=args["explore"]+args["observe"])

    #Optimizer update :
    args.update({
        "scheduler" : True,
        "scheduler_rate" : 0.99,
        "scheduler_type" : "exponential",
        "lr" : 0.00005,
        "momentum" : 0.9,
        "seed" : 1,
        "use_cuda" : torch.cuda.is_available(),
        "optimizer" : "Adam", # 
        #"optimizer" = "SGD"
    })
    