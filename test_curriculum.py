from rocket_env import RocketLandingEnv, STAGES
import numpy as np

for s in range(1, 7):
    env = RocketLandingEnv(stage=s)
    obs, _ = env.reset(seed=0)
    action = env.action_space.sample()
    obs2, r, term, trunc, info = env.step(action)
    pad = STAGES[s]["pad"]
    adim = env.action_space.shape[0]
    print(f"Stage {s}: action_dim={adim}  obs_dim={len(obs)}  pad={pad}m  OK")

print("\nAll stages passed.")
