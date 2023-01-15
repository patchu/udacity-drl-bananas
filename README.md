# udacity-drl-bananas

Project submission for Project 1 of the Udacity Deep Reinforcement Learning class: Bananas

## Project Details

This repository contains a solution for Project 1 of the Udacity Deep Reinforcement Learning project. The goal is to train an agent through reinforcement learning to navigate through a modified version of the **Banana collector** environment from Unity ML Agents: a simulated courtyard containing yellow and blue bananas, with a goal of gathering as many yellow bananas as possible while avoiding blue bananas.

## Requirements

### Set up Python

This project was written and tested on an AWS EC2 instance type of `m6i.xlarge` running the DLAMI based on Ubuntu 20.04, although it should work on any Intel-based computer.

With `anaconda` or `miniconda` already installed on your computer, type:

```bash
conda create -n bananas python=3.8
conda activate bananas

pip install .
```

This will create a Python environment for you and install all required dependencies.

**CAVEATS**

* The `requirements.txt` file is modified from the original Udacity project. Using the file in the original Udacity project resulted in version conflicts, and this is my best guess at a combination of Python package versions that still work.


### Install the Bananas app

Install the Bananas app that works with your OS:

* [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
* [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
* [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
* [Windows (64 bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

Download and unzip this file in the root directory of this project.

### Start the Jupyter notebook

In the root directory of this project, type:

```bash
jupyter notebook
```

Then open the file `Bananas.ipynb` to train the agent. Further instructions are in the `Bananas.ipynb` file itself

### Files

This project contains the following files:

* **Bananas.ipynb**: The main Jupyter notebook that contains code to train the agent
* **dqn_agent.py**: An implementation of a trainable agent which uses two DQN networks
* **model.py**: A model of a single DQN agent, implemented as 2 or 3 fully-connected Dense neural network layers
* **README.md**: The file you're currently reading
* **requirement.txt**: A modified version of the original requirements.txt, listing all the versions of all the Python libraries required to get this project working
* **setup.py** Used with `requirements.txt` to set up the project

Fastest solve files:

* **Fastest.ipynb**: Notebook with the results of the fastest agent solve
* **fastest_checkpoint.pth**: the `checkpoint.pth` file for the fastest agent solve

Visual agent training:

* **Bananas_Pixels.ipynb**: The main Jupyter notebook that runs the raw pixel agent training
* **dqn_agent_visual.py**: The DQN agent for raw pixel training
* **model_pixel.py**: 2 or 3 block CNN to support DQN agent for raw pixel training

AWS Trainium:

* **dqn_agent_trainium.py**: The AWS Trainium GPU needs an addition Pytorch library and a line added to the optimization step, so I created another file to facilitate switching between Trainium and other GPUs



# More information

For more information about this project, please continue to the
[**Report**](./Report.md)

