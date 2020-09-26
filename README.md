# Udacity Project: Continuous Control

## Problem
The Unity ML-agents Reacher Environment with a single agent is considered. The agent is a double-jointed arm that can move to target locations. The agent receives a reward of +0.1 for each step when the agent's hand is in the goal location. The goal of the agent is to maintain its position at the target location as many steps as possible. This problem is considered to be solved if the agent receives an average score of +30 over 100 consecutive episodes.

The observation space includes 33 variables consisting of position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four real values that correspond to torque applicable to two joints. Every entry in the action vector should be a value between -1 and 1.

## Getting Started
To set up your python environment to run the code in this repository, follow the instruction
in [Deep Reinforcement Learning Nanodegree](https://github.com/udacity/deep-reinforcement-learning#dependencies)

To install the Reacher environment of a single agent (Version 1), follow the instruction in [Here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control)

## Instructions
Before running the following commands, add the unity Reacher environment filename to each file.

The agent is trained by the following:

python ./main_train_v1.py

To visualize the trained agent, run the following:

python ./visualize_agent_v1.py

## Report
[Here is a report](Report.md)