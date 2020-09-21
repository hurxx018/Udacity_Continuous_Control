# Report

## Implementation
The agent employs a deep deterministic policy gradient (DDPG) algorithm that consists of Actor and Critic. The Actor consists of two fully-connected hidden layers (See the Architecture below), and the Critic includes three fully-connected hidden layers (See the Architecture below). The two actors and the two critics are used in the agent. For the Actors and the Critics, one is named by local and the other is called as target. The agent utilizes the target network to update the local network, which is similar to the DQN algorithm. The soft-update strategy is used with TAU. 

During the training, the target Actor and Critic networks are updated periodically by employing the experiences stored in the replay buffer or the agent's memory. The replay buffer has a size of 200000. The periodicity is defined by a parameter UPDATE_EVERY in ddpg_agent.py. When the network is updated, the learning is repeated by N_LEARNING. The number of experiences used for the learning is defined by BATCH_SIZE in ddpg_agent.py. During the training, noise is added to the action value. The noise is modeled by Ornstein–Uhlenbeck noise process. 

## Learning Algorithm
The agent utilizes the Deep Deterministic Policy Gradient (DDPG).

## Hyperparameters
BATCH_SIZE = 64
TAU = 0.01
UPDATE_EVERY = 10
N_LEARNING = 4

Learning rates for the Actor and Critic are given by 
LR_ACTOR = 5e-4
LR_CRITIC = 5e-4

Here are parameters of Ornstein–Uhlenbeck noise process.
THETA = 0.01
SIGMA = 0.005


## Architecture
The Actor network consists of an input layer, two hidden layers (FC1 and FC2), and one output layer. Fully connected layers were used with leakyReLU activation function with a negative slope of 0.2. Batch normalization is applied between FC1 and FC2.
FC1 and FC2 layers includes 256 and 512 nodes, respectively.

![figure of architecture](https://github.com/hurxx018/Udacity_Continuous_Control/blob/master/images/Actor.png)

The Critic network consists of an input layer of state space and action space, three hidden layers, and one output layer. Action space is concatenaed to the output of the first hidden layer denoted by FCS1 with 128 nodes. The next two hidden layers have 128 and 64 nodes respectively. LeakyReLu was used as activation function with a negative slope of 0.2

![figure of architecture](https://github.com/hurxx018/Udacity_Continuous_Control/blob/master/images/Critic.png)


## Results
This is a trace of mean values of 100 consecutive scores. The agent achieves the goal after running ~290 episodes. The initial stage of learning is steep and the learning phase becomes slow down. 

![figure of score](https://github.com/hurxx018/Udacity_Continuous_Control/blob/master/images/score.png)


## Future Work
This work needs to be improved with the implementation of prioritized experience replay.