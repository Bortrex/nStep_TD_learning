# One-Step vs. $n$-Step TD Control in Taxi-v3

This project explores how one-step and multi-step temporal-difference (TD) learning algorithms perform in the Gymnasium Taxi-v3 environment.

Three tabular reinforcement learning algorithms are implemented and compared: Q-learning, 4-step Q-learning, and 4-step SARSA. The experiment examines how the choice of learning algorithm and the use of multi-step returns affect the agents' learning behavior and accumulated rewards.

Each algorithm is trained over multiple runs, and the resulting learning curves are used to compare their performance during training.

## Usage

Install Python with `gymnasium`, `numpy`, `matplotlib`, and Jupyter. From the repository directory, open the notebook and run all cells in order:

```bash
jupyter notebook gym_taxiV3.ipynb
```

The notebook also saves the two comparison figures in `plots/`.

## Agents

### Q-learning

Q-learning is a model-free algorithm that updates action values after each transition.

$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$

### 4-Step Q-learning

This variant uses four observed rewards before bootstrapping from the maximum action value.

$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [G_{t:t+n} - Q(s_t, a_t)]$

With $n = 4$:

$G_{t:t+n} = r_{t+1} + \gamma r_{t+2} + \ldots + \gamma^{n-1} r_{t+n} + \gamma^n \max_{a'} Q_{t+n-1}(s_{t+n}, a')$

### 4-Step SARSA

SARSA (State-Action-Reward-State-Action) uses the same four-reward horizon, but bootstraps from the action selected by its epsilon-greedy policy.

$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [G_{t:t+n} - Q(s_t, a_t)]$

With $n = 4$:

$G_{t:t+n} = r_{t+1} + \gamma r_{t+2} + \ldots + \gamma^{n-1} r_{t+n} + \gamma^n Q_{t+n-1}(s_{t+n}, a_{t+n})$

These returns stop without bootstrapping at true termination. At the 300-step time limit, pending returns instead bootstrap from the final nonterminal state, using the appropriate action value for each algorithm.

## Results

The Taxi-v3 experiment uses 500 episodes per run and 10 runs with seeds 123–132. Hyperparameters are fixed: learning rate $\alpha=0.15$, discount factor $\gamma=0.95$, $\epsilon$-greedy=0.2, and a 300-step episode limit. Both multi-step methods use $n = 4$.

Each curve is the mean exponentially smoothed episode reward across the 10 runs. Within each run, smoothing starts at zero and follows $M_e = 0.95 M_{e-1} + 0.05 R_e$, where $R_e$ is the total reward in episode $e$.

![Zoomed learning curves comparing Q-learning, 4-Step Q-learning, and 4-Step SARSA](plots/nStep_TD_learning_zoomed.png)

*Main comparison: the reward-axis zoom emphasizes later training performance. Values below −200 are clipped; the horizontal axis still spans all 500 episodes.*

<img src="plots/nStep_TD_learning.png" alt="Full learning curves showing the early reward decline and subsequent improvement over 500 episodes" width="520">

Q-learning has higher mean smoothed reward early in training. Around the middle, 4-Step Q-learning moves ahead. 4-Step SARSA improves more strongly later and moves above Q-learning. Near the end, 4-Step Q-learning reaches the highest mean smoothed reward, followed by 4-Step SARSA and Q-learning (approximately −16.53, −23.26, and −38.98 at the final episode).

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Author

– [@Bortrex](https://github.com/Bortrex)
