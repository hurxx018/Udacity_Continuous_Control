# Report

## Implementation
The agent utilizes a Deep Deterministic Policy Gradient (DDPG) algorithm that consists of the Actor and the Critic.  The Actor and the Critic are defined by neural networks (See Architectures for the Actor and the Critic below for the details).  The agent uses the two actors and the two critics. For the two actors, the one is named by local and the other is called as target.  This naming is the same for the two critics.  The agent utilizes the target networks to update the local network, which is similar to the Deep Q-network (DQN) algorithm.  For a given state, the agent makes a decision of action by using its local actor to interact with the Reacher environment. 

![Figure of Perception-Action Cycle for the DDPG Agent](https://github.com/hurxx018/Udacity_Continuous_Control/blob/master/images/DDPG_Agent_percetion_action_cycles.png)

## Learning Algorithm
The agent includes local and target actors and local and target critics.  Each experience is stored as a tuple of state, action, reward, next state, and done in the replay buffer.  The agent samples previous experiences of BATCH_SIZE from the replay buffer at the learning stage.  The learning stage occurs periodically.  The periodicity is defined by a parameter UPDATE_EVERY.  At each learning stage, the learning repeats N_LEARNING times.  (BATCH_SIZE, UPDATE_EVERY, and N_LEARNING are defined in ddpg_agent.py.)

The local actor and critic are updated through the optimization with two Adam optimizers.  The one optimizer is used for the actor and the other is for the critic.  The local critic estimates the Q-value of a given state and action pair.  This estimated action value is compared with the value obtained from the target critic of the next state and the next action.  The next action values are determined by the target actor for a given next state.  The sum of reward and the discounted Q-value of the target critic from the next state and the next action plays a role of expected the Q value. The error between the estimated Q-value and the expected Q-value is used to update the parameters of local critic with the mean-squared-error loss.  The target critic is like a model of Q value for the environment and the local critic plays a role of learning Q table for the environment. 

The local critic evaluates the local actor by calculating the Q-value for a state and an action determined by the local actor.  The optimizer updates the parameters of the local actor in a direction of increasing the Q-values given by the local critic. 

Once updating the local actor and critic, the target actor and critic are updated from the local actor and critic by the soft-update with a parameter of TAU. The TAU is a weight factor for the local parameters on averaging the local parameters and the target parameters for the target actor and critic. 

![Figure of Learning Process for the DDPG Agent](https://github.com/hurxx018/Udacity_Continuous_Control/blob/master/images/DDPG_learning_process.png)

During the training, noise is added to the action values.  The noise is modeled by Ornstein–Uhlenbeck noise process with two parameters THETA and SIGMA.  The value of THETA is reduced by a factor DECAY_FACTOR_S at the end of each episode, and SIGMA is also reduced by DECAY_FACTOR_T. THETA and SIGMA are set relatively large at the begin of the training which helps the agent to explore the environment.  The two parameters were reduced to stabilize the decision of action values as the agent accumulates the experiences while passing through multiple episodes.

## Summary of Hyperparameters
This project uses a number of experiences given by BATCH_SIZE = 64.

The parameters of the Actor and the Critic are updated by the soft-update with a fraction given by TAU = 0.01

Here are the parameters of the learning periodicity and the repetition: UPDATE_EVERY = 10 and N_LEARNING = 4

Learning rates for the Actor and Critic are given by LR_ACTOR = 5e-4 and LR_CRITIC = 5e-4, respectively.

Here are parameters of Ornstein–Uhlenbeck noise process: THETA = 0.01, SIGMA = 0.005, DECAY_FACTOR_S = 0.999, and DECAY_FACTOR_S = 0.999


## Architectures for the Actor and the Critic
The Actor network consists of an input layer, two hidden layers (FC1 and FC2), and one output layer.  The FC1 and FC2 layers include 256 and 512 nodes, respectively.  They are used with leakyReLU activation function with a negative slope of 0.2.  Batch normalization (BN1) is applied to the output of the layer FC1.  The activation function of the output layer is hyperbolic-tangent that ensures the output values of the range from -1 to 1. 

![Figure of Actor architecture](https://github.com/hurxx018/Udacity_Continuous_Control/blob/master/images/Actor.png)

The Critic network includes an input layer, three hidden layers (FCS1, FC2, FC3), and an output layer.  The state space of 33 variables is converted  to a layer of 128 nodes with the relu activation. input layer has 33 nodes for state space that are activated by leaky relu with a negative slope of 0.2. The output of the input layer is concatenated with the action values, that is the 

 
 Action space is concatenated to the output of the first hidden layer denoted by FCS1 with 128 nodes. The next two hidden layers have 128 and 64 nodes respectively. LeakyReLu was used as activation function with a negative slope of 0.2

![Figure of Critic architecture](https://github.com/hurxx018/Udacity_Continuous_Control/blob/master/images/Critic.png)


## Results
This is a trace of mean values of 100 consecutive scores. The agent achieves the goal after running ~290 episodes. The initial stage of learning is slow to achieve about 5 during the first 100 episodes. The leaning becomes steeper during the next 100 episodes. The agent achieve a score of ~37 after 250 episodes. 

![Figure of Score](https://github.com/hurxx018/Udacity_Continuous_Control/blob/master/images/score.png)


## Future Work
This work needs to be improved with the implementation of prioritized experience replay.