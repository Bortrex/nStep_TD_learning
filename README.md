# One-Step vs. $n$-Step TD Control in Taxi-v3

This project explores how one-step and multi-step temporal-difference (TD) learning algorithms perform in the Gymnasium Taxi-v3 environment.

Three tabular reinforcement learning algorithms are implemented and compared: Q-learning, 4-step Q-learning, and 4-step SARSA. The experiment examines how the choice of learning algorithm and the use of multi-step returns affect the agents' learning behavior and accumulated rewards.

Each algorithm is trained over multiple runs, and the resulting learning curves are used to compare their performance during training.



## Agents

### QLearning

QLearning is a model-free RL algorithm that learns the value of an action in a particular state. It does not require a model of the environment.

- Update rule:
  
    $Q(s, a) \leftarrow Q(s, a) + \alpha \left[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right]$

### N-Step QLearning
N-step QLearning is again a variant of QLearning that also looks n-steps ahead before updating the value estimates.

- Update rule:

    $Q(s_t, a_t) \gets Q(s_t, a_t) + \alpha \\: [G_{t:t+n} - Q(s_t, a_t)]$
  
  with: $G_{t:t+n} = r_{t+1} + \gamma \\: r_{t+2} + \ldots + \gamma^{n-1} \\: r_{t+n} + \gamma^n \\: \max_{a'} Q_{t+n-1} \\: (s_{t+n}, a')$

### N-Step SARSA
N-step SARSA is an extension of the standard SARSA (State-Action-Reward-State-Action) learning algorithm, using the idea of looking n-steps ahead in the action-value updating rule. This approach helps in accelerating the learning process by leveraging more future information.

- Update rule:

    $Q(s_t, a_t) \gets Q(s_t, a_t) + \alpha \\: [G_{t:t+n} - Q(s_t, a_t)]$
  
  with: $G_{t:t+n} = r_{t+1} + \gamma \\: r_{t+2} + \ldots + \gamma^{n-1} \\: r_{t+n} + \gamma^n \\: Q_{t+n-1} \\: (s_{t+n}, a_{t+n})$




## Results
Results from our experiments indicate how each algorithm performs in terms of cummulative reward across various episodes.

![cumm_reward](https://github.com/Bortrex/TD_learning/assets/24497590/286a1b95-8b2d-4f9b-b4fd-41faeba4b759)

### N-Step QLearning

The best performing agent in the environment is $n$-step Q Learning algorithm. Below is the resulting model.

![QL_gif](https://github.com/user-attachments/assets/e9ac9317-1fb0-4ec8-ace1-1161e5115c2a)

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Authors

– [@Bortrex](https://github.com/Bortrex)


