# udacity-drl-bananas

Project submission for Project 1 of the Udacity Deep Reinforcement Learning class: Bananas

## Project Details

This repository contains a solution for Project 1 of the Udacity Deep Reinforcement Learning project. The goal is to train an agent through reinforcement learning to navigate through a modified version of the *Banana collector* environment from Unity ML Agents: a simulated courtyard containing yellow and blue bananas, with a goal of gathering as many yellow bananas as possible while avoiding blue bananas.

## Requirements

### Set up Python

This project was written and tested on an AWS EC2 instance type of m6i.xlarge running the DLAMI, although it should work on any Intel-based computer.

With `anaconda` or `miniconda` installed on your computer, type:

```bash
conda create -n bananas python=3.6
conda activate bananas

pip install .
```

This will create a Python environment for you and install all required dependencies.

*CAVEATS*

* The `requirements.txt` file is modified from the original Udacity project. Using the file in the original Udacity project resulted in version conflicts, and this is my best guess at a set of version that still work.
* The custom-built Unity environment will not work on a Mac/Macbook M1 -- I can personally verify this
* Python 3.6 seems to be the preferred environment for the Bananas app, since the app appears to be over 4 years old, and relies on several outdated Python modules.

### Install the Bananas app

Install the Bananas app that works with your OS:

* [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
* [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
* [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
* [Windows (64 bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

Download and unzip this file in the root directory of this project. Further instructions are in the `Bananas.ipynb` file itself
