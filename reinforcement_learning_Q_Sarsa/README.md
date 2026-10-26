# Reinforcement Learning: Q-Learning and SARSA

## Intuition
Learn action values for a racing environment with stochastic failures and directional dynamics.

## Core updates
- Q-learning: `Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]`
- SARSA: `Q(s,a) ← Q(s,a) + α [r + γ Q(s',a') - Q(s,a)]`

## Theory from First Principles
### 1) Goal
Learn a policy that navigates the racetrack to finish quickly while avoiding crashes.

### 2) Approach
State includes position, speed, and direction. Actions choose heading plus acceleration. Environment is stochastic: acceleration can fail and sharp turns can fail with probability depending on speed/turn angle. Training uses `\epsilon`-greedy exploration with decaying `\epsilon`, and temporal-difference updates for Q-learning (off-policy) and SARSA (on-policy).

### 3) Result / Output
Training returns a Q-table and episode returns. `run(...)` executes greedy policy from a random start to generate a trajectory.

### 4) Formulas
- Bellman target (Q-learning): `r+\gamma\max_{a'}Q(s',a')`
- Bellman target (SARSA): `r+\gamma Q(s',a')` where `a'` is the next chosen action
- Update used in code: `Q(s,a)\leftarrow Q(s,a)+\alpha_t\,[\text{target}-Q(s,a)]`
- Here `\alpha_t = 1/(1+0.01\,N(s,a))` (visit-count adaptive), even though constructor accepts `alpha`.

Variables: `\gamma` discount, `r` immediate reward (`+20` finish, `-2` crash reset, `-1` otherwise), `N(s,a)` visit count.

### 5) Intuition (small example)
If a fast turn frequently crashes, repeated negative rewards lower its Q-value. Safer sequences that still reach finish accumulate better long-term return, so greedy behavior gradually shifts toward them.


## Files
- `model.py`: environment, state/action encoding, Q-learning/SARSA loops.
- `tracks.py`: track definitions and loader.
- `main.py`: long-run training, curve summary, path visualization.

## From-scratch vs libraries
- From-scratch environment + RL updates.
- Matplotlib used for plotting outputs.

## Dependencies
- `numpy`, `matplotlib`

## Usage
```bash
python reinforcement_learning_Q_Sarsa/main.py
```

## Dataset
- None (tracks encoded in code).

## Limitations / caveats
- Default script uses 50,000 episodes per algorithm and can be slow.
- Constructor parameter `alpha` is currently not directly used in TD update; update uses visit-count adaptive rate.
