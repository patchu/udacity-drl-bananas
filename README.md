# udacity-drl-bananas

Project submission for Project 1 of the Udacity Deep Reinforcement Learning class: Bananas

## Project Details

This repository contains a solution for Project 1 of the Udacity Deep Reinforcement Learning project. The goal is to train an agent through reinforcement learning to navigate through a modified version of the **Banana collector** environment from Unity ML Agents: a simulated courtyard containing yellow and blue bananas, with a goal of gathering as many yellow bananas as possible while avoiding blue bananas.

## Requirements

### Set up Python

This project was written and tested on an AWS EC2 instance type of `m6i.xlarge` running the DLAMI based on Ubuntu 20.04, although it should work on any Intel-based computer.

With `anaconda` or `miniconda` already installed on your computer, type:

```bash
conda create -n bananas python=3.6
conda activate bananas

pip install .
```

This will create a Python environment for you and install all required dependencies.

**CAVEATS**

* The `requirements.txt` file is modified from the original Udacity project. Using the file in the original Udacity project resulted in version conflicts, and this is my best guess at a combination of Python package versions that still work.
* Python 3.6 seems to be the preferred environment for the Bananas app, since the app appears to be over 4 years old, and relies on several outdated Python modules.

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


