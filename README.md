# Google-Research-Football

# Install :

### Kaggle environments.

`git clone https://github.com/Kaggle/kaggle-environments.git`

`cd kaggle-environments && pip install .`

### GFootball environment.

`apt-get update -y`

`apt-get install -y libsdl2-gfx-dev libsdl2-ttf-dev`

### Make sure that the Branch in git clone and in wget call matches !!

`git clone -b v2.6 https://github.com/google-research/football.git`

`mkdir -p football/third_party/gfootball_engine/lib`

`wget https://storage.googleapis.com/gfootball/prebuilt_gameplayfootball_v2.6.so -O football/third_party/gfootball_engine/lib/prebuilt_gameplayfootball.so`

`cd football && GFOOTBALL_USE_PREBUILT_SO=1 pip3 install .`

### Other Dependencies :

    Pytorch, Numpy, Matplotlib, Pygame

# TRAIN :

## Training from scratch

Run a train experiment on a notebook :

`python run_exp.py --experiment_name $EXPERIMENT_NAME --running_in_notebook`

In this case, models and parameters will be saved in `/content/drive/My Drive/google-football/$EXPERIMENT_NAME/`

One can change the default path using :

`--global_path $path`

In this case, models and parameters will be saved in `$global_path/$EXPERIMENT_NAME/`

## When loading a model

`python run_exp.py --experiment_name $EXPERIMENT_NAME --running_in_notebook --load_model`

In this case, we will load the latest model in `/content/drive/My Drive/google-football/$EXPERIMENT_NAME/`

One can change the loaded iteration using `--t`
