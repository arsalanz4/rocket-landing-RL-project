# Rocket Landing RL — PPO Agent Trained from Scratch

A custom reinforcement learning environment built from scratch to train an autonomous rocket landing agent using Proximal Policy Optimisation. Everything in this project was written independently — the physics engine, the reward function, the curriculum, and the training pipeline.

---

## What This Is

The agent controls a 2D rocket and must learn to land it softly on a target pad. It controls two things: how much thrust to apply and how to steer using a gimbal. The challenge is that it has to manage four competing goals at the same time — slow down vertically, navigate to the pad horizontally, maintain an upright angle, and not run out of fuel.

There is no pre-built simulation. The physics, the reward signal, and the learning environment were all designed from the ground up.

---

## How Training Works

Rather than throwing the full difficulty at the agent immediately, training uses a curriculum — a sequence of progressively harder stages. The agent must achieve at least 80 percent success across three consecutive evaluation windows before advancing to the next stage. Each stage introduces one new challenge, such as a smaller landing pad, higher starting altitude, faster initial descent, reduced attitude control assistance, or wind gusts.

This approach means failure modes can be isolated and diagnosed cleanly. When the agent gets stuck, it is usually because one specific new challenge is too hard given the current policy — not because everything is wrong at once.

---

## Project History

### Version 1 — 8D Environment

The original environment used an 8-dimensional observation space covering position, velocity, angle, angular velocity, fuel, and throttle. After extensive training and several rounds of reward shaping and curriculum design fixes, the agent reached 90 percent success at stage 7 with near-perfect landing precision. Vertical velocity at touchdown was consistently between -4.0 and -4.5 metres per second, horizontal velocity under 1.0 metres per second, and landing position within 0.8 metres of the pad centre.

This version is documented in the commit history and represents a complete, working result.

### Version 2 — 9D Environment (Current)

The current environment adds a ninth observation — horizontal wind force — and extends the curriculum to 12 stages including wind gusts up to 20 metres per second squared. The agent is currently relearning the combined skill set through the curriculum after a series of significant design improvements. Key improvements over version 1 include a log standard deviation clamp to prevent action distribution collapse, a bidirectional horizontal velocity correction reward, curriculum-based objective isolation to prevent vertical reward signals from drowning out horizontal learning, and paired VecNormalize checkpoint saving to prevent normalisation mismatch on rollback.

---

## Key Technical Challenges

Five significant failure modes were encountered and resolved during development. A full technical writeup is available as a research paper linked below.

**Unbounded action distribution variance.** PPO does not constrain the absolute value of log standard deviation, only the per-update change. Over millions of steps, the action std grew to astronomical values and collapsed the policy. Fixed with a hard clamp callback.

**Thrust-to-weight resource contention.** At the operating fuel load, the rocket has a thrust-to-weight ratio of approximately 1.003, barely hover-capable. Adding fuel to increase margin actually worsens the situation because it adds mass against a fixed engine thrust. The curriculum had to be restructured around this physical constraint.

**Asymmetric reward signal.** The original horizontal velocity correction only rewarded moving toward the target velocity, contributing zero when moving away. This created a stable local optimum where the agent drifted freely in the wrong direction. Fixed by making the correction bidirectional so that wrong-direction drift is actively penalised.

**Direction-blind velocity lock.** Even after fixing the asymmetry, the agent locked onto a constant horizontal velocity ignoring position entirely. The root cause was that vertical reward signals dominated the advantage estimates, making the horizontal gradient invisible during updates. Fixed by inserting a diagnostic stage where vertical difficulty was trivially easy, isolating horizontal control learning.

**VecNormalize checkpoint mismatch.** Loading model weights without their paired normalisation statistics caused subtle policy degradation after rollbacks, misattributed for some time to reward or curriculum issues. Fixed by saving VecNormalize statistics alongside every model checkpoint.

---

## Results

**Version 1 (8D, stage 7):** 90 percent success rate. Consistent soft landings within 0.8 metres of pad centre. Vertical velocity at touchdown -4.0 to -4.5 metres per second. Horizontal velocity under 1.0 metres per second.

**Version 2 (9D, in progress):** Currently relearning through curriculum stages after environment upgrade and design improvements. Horizontal velocity lock resolved. Wind stages not yet reached.

---

## Tech Stack

- Python
- Stable-Baselines3 — PPO implementation
- Gymnasium — environment interface
- PyTorch — neural network backend
- pygame-ce — visual renderer
- NumPy — physics calculations

---

## Research Paper

A detailed technical writeup covering all five failure modes and their implications for reinforcement learning in multi-objective continuous control problems is available here: 

---

## Author

Arsalan Zadran 

Law graduate, AI developer, Geospatial intelligence researcher.

GitHub: github.com/arsalanz4
