# Google-Research-Football

# Install :

### Kaggle environments.
```git clone https://github.com/Kaggle/kaggle-environments.git```

```cd kaggle-environments && pip install .```

### GFootball environment.
```apt-get update -y```

```apt-get install -y libsdl2-gfx-dev libsdl2-ttf-dev```

### Make sure that the Branch in git clone and in wget call matches !!
``` git clone -b v2.6 https://github.com/google-research/football.git```

``` mkdir -p football/third_party/gfootball_engine/lib```

```wget https://storage.googleapis.com/gfootball/prebuilt_gameplayfootball_v2.6.so -O football/third_party/gfootball_engine/lib/prebuilt_gameplayfootball.so```

```cd football && GFOOTBALL_USE_PREBUILT_SO=1 pip3 install .```

### Other Dependencies :
    Pytorch, Numpy, Matplotlib, Pygame
