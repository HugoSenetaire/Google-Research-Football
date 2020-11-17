import argparse
import torch


def create_args():
    parser = argparse.ArgumentParser(description='DFP exp')
    parser.add_argument('--gpu', action='store_true', help='Whether to use GPU')
    args = parser.parse_args()
    return vars(args)



def update_default_args(args):
    # Games update :
    args.update({
        "CHANNEL_NAMES" : ["left_team", 'right_team', 'left_team_direction', 'right_team_direction', 'ball', 'ball_direction'],
        "MEASUREMENT_NAMES" : ['goals', 'ball_distance_to_goal', "ball_distance_to_center", "possession"],
        "TIMESTEPS" : [1,2,4,8,16,32],
        "IMAGE_SIZE" : (43, 101),
        "NB_ACTIONS" : 19,
    })

    # DFP Agent update :
    args.update({
        "lambda_relu" : 0.05,
        "gamma" : 0.99,
        "epsilon" : 1.0,
        "initial_epsilon" : 1.0,
        "batch_size" : 32,
        "explore" : 100000,
        "observe" : 2000,
        "frame_per_action" : 4,
        "timestep_per_train" : 5,
        "max_memory" : 20000
    })


    #Optimizer update :
    args.update({

        "lr" : 0.00005,
        "momentum" : 0.9,
        "seed" : 1,
        "use_cuda" : torch.cuda.is_available(),
        "optimizer" : "Adam" # 
        #"optimizer" = "SGD"
    })
    