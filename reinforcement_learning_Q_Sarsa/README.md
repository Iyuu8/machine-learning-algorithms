# Reinforcement Learning: Q-Learning and SARSA

## Intuition
Learn action values for a racing environment with stochastic failures and directional dynamics.

## Core updates
- Q-learning: `Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]`
- SARSA: `Q(s,a) ← Q(s,a) + α [r + γ Q(s',a') - Q(s,a)]`

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
