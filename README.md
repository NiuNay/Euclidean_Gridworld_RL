# RL for navigation

## Overview
This is a repo for training reinforcement-learning agents is a gridworld, such that agents optimize for the shortest integrated Euclidean distance to rewards. Currently supported models:
- Tabular Q learning
- State-action Successor Representation
- Model-based tree search

There are files in this repository corresponding to different models but these are not currently functional.

## Usage
SETUP:
- Create a virtual environment and activate it:\
```conda create --name [YOUR_ENV_NAME] python=3.8```\
```conda activate [YOUR_ENV_NAME]```
- Prepare the repo with the following commands:\
```git clone https://github.com/NiuNay/Euclidean_Gridworld_RL.git```\
```cd Euclidean_Gridworld```\
```pip install -r requirements.txt```\
```pip install -e .```

SUBMITTING JOBS:
- Navigate to the experiments folder using: ```cd rl_nav/experiments```
- Modify the config files as needed:
  - The base config file for each model can be found in experiments/suites/\[MODEL_NAME]/config.yaml\
    - If you want to train a model using data from a file, add your own to the rl_nav/training_data folder or unzip training_data_zipped.zip to find a ready-to-use selection
    - The "code" folder in the same directory contains Jupyter Notebooks that can be used to produce more random walk data files or process the real mouse data (condition_1.pkl) 
  - An additional file found under experiments/suites/config_changes.yaml can be used to run the simulation multiple times, changing condition (different gridworld training map) each time
    - There are 4 main conditions, outlined in Shamash et al.'s 2023 paper: https://doi.org/10.1016/j.neuron.2023.03.034
    - Simply uncomment the conditions in which you would like the simulations to run on
- Run an experiment, e.g. with the command ```python run.py --mode parallel --seeds 2 --config_path suites/[MODEL_NAME]/config.yaml --config_changes config_changes.py```


VISUALIZING RESULTS: use the command ```python post_process.py --results_folder results/[RESULTS_FOLDER_NAME] --plot_trajectories```
