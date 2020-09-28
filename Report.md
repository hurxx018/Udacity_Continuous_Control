# Report

## Implementation
The agent utilizes a deep deterministic policy gradient (DDPG) algorithm that consists of the Actor and the Critic. The Actor and the Critic are defined by neural networks (See Architectures for the Actor and the Critic below for the details). The agent uses the two actors and the two critics. For the two actors, the one is named by local and the other is called as target. This naming is the same for the two critics. The agent utilizes the target networks to update the local network, which is similar to the Deep Q-network (DQN) algorithm. For a given state, the agent makes a decision by using a local actor. 

![Figure of Perception-Action Cycle for the DDPG Agent](https://github.com/hurxx018/Udacity_Continuous_Control/blob/master/images/DDPG_Agent_percetion_action_cycles.png)

During the training, the target Actor and Critic networks are updated periodically by employing the experiences stored in the replay buffer or the agent's memory. The replay buffer has a size of 200000. The periodicity is defined by a parameter UPDATE_EVERY in ddpg_agent.py. When the network is updated, the learning is repeated by the number of times given by N_LEARNING. The number of experiences used for the learning is defined by BATCH_SIZE in ddpg_agent.py. 

Two Adam optimizers are used for
The soft-update strategy is used with TAU. 

During the training, noise is added to the action values.  The noise is modeled by Ornstein–Uhlenbeck noise process with two parameters THETA and SIGMA.  The value of THETA is reduced by a factor DECAY_FACTOR_S at the end of each episode, and SIGMA is also reduced by DECAY_FACTOR_T. THETA and SIGMA are set relatively large at the begin of the training which helps the agent to explore the environment.  The two parameters were reduced to stabilize the decision of action values as the agent accumulates the experiences while passing through multiple episodes.


## Learning Algorithm
The agent utilizes the Deep Deterministic Policy Gradient (DDPG) consisting of the Actor and the Critic. The Actor and the Critic are defined by the neural networks in this project. 




The Actor consists of two fully-connected hidden layers (See the Architecture below for the details) and its outputs are three continuos values between -1 and 1. The Critic includes three fully-connected hidden layers (See the Architecture below) and its output is a single continuous value. The Critic depends on both the state and the output of the Actor and evaluates the decision made by the Actor.

Both the Actor and the Critic are updated by using the two Adam optimizers that are for each network respectively. Learning rates for both of 

## Summary of Hyperparameters
This project uses a number of experiences given by 

BATCH_SIZE = 64

and the parameters of the Actor and the Critic are updated by the soft-update with a fraction given by 

TAU = 0.01

UPDATE_EVERY = 10

N_LEARNING = 4

Learning rates for the Actor and Critic are given by 

LR_ACTOR = 5e-4

LR_CRITIC = 5e-4

Here are parameters of Ornstein–Uhlenbeck noise process.

THETA = 0.01

SIGMA = 0.005


## Architectures for the Actor and the Critic
The Actor network consists of an input layer, two hidden layers (FC1 and FC2), and one output layer.  Fully connected layers are used with leakyReLU activation function with a negative slope of 0.2.  Batch normalization (BN1) is applied to the output of the layer FC1.  The FC1 and FC2 layers include 256 and 512 nodes, respectively.  The activation function of the output layer is hyperbolic-tangent that ensures the output values of the range from -1 to 1. 

![Figure of Actor architecture](https://github.com/hurxx018/Udacity_Continuous_Control/blob/master/images/Actor.png)

The Critic network includes an input layer, three hidden layers (FCS1, FC2, FC3), and one output layer.  The input layer has 33 nodes for state space. 

 
 Action space is concatenated to the output of the first hidden layer denoted by FCS1 with 128 nodes. The next two hidden layers have 128 and 64 nodes respectively. LeakyReLu was used as activation function with a negative slope of 0.2

![Figure of Critic architecture](https://github.com/hurxx018/Udacity_Continuous_Control/blob/master/images/Critic.png)


## Results
This is a trace of mean values of 100 consecutive scores. The agent achieves the goal after running ~290 episodes. The initial stage of learning is slow to achieve about 5 during the first 100 episodes. The leaning becomes steeper during the next 100 episodes. The agent achieve a score of ~37 after 250 episodes. 

![Figure of Score](https://github.com/hurxx018/Udacity_Continuous_Control/blob/master/images/score.png)


## Future Work
This work needs to be improved with the implementation of prioritized experience replay.