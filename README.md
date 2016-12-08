# car_racing_deep_RL

Prerequisites
-------------
Tensorflow 0.90
OpenAI gym

# Note!
For REINFORCE method experiment in the report, swithch to the `time_step_base_reward` branch
and load the trained model `norview.gamma0.999-1655` by typing:
`vglrun ipython nn_car_racing.py -- -log tmplog.log -chkpt norview.gamma0.999-1655`

For supervised learning experiments in the report, switch to the `t_s_b_r_suplearn_3dAction_only3steerAction` branch
and run `vglrun ipython nn_car_racing.py -- -log tmplog.log`

For DDPG experiments in the report, switch to the `ddpg` branch
and run `vglrun ipython ddpg.py`

In any case, don't run the files in master branch, files are mixed in master branch

# Main files

nn_car_racing.py
-----------------
The `main` file that starts the video game

lib.py
---------------------------
contains the network definition and the agent

ddpg.py
-----------------
a separate file for DDPG experiment 
it is a counterpart of both nn_car_racing.py and lib.py because it
both starts the video game and contains the network definition and the agent


# Other files

datagen_car_racing.py
-----------------------
generate human play data for supervised learning

SL_pretrain.py
------------------
Train the policy network defined in lib.py using human play data, and dump the trained model.

human_car_racing.py
-----------------------
play this video game by yourself

my_car_env.py
--------------------
I modified the car-racingv0 environment's source code, and it is stored here.
Actually all the files in this repository and all the experiments done use this environment instead of the original car-racing environment.
