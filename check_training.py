import numpy as np

data = np.load('rocket/logs/evaluations.npz')
timesteps = data['timesteps']
results = data['results']
ep_lengths = data['ep_lengths']

for i in range(len(timesteps)):
    mean_r = results[i].mean()
    mean_l = ep_lengths[i].mean()
    print(f"  step {timesteps[i]:>8,}  |  mean reward: {mean_r:+.1f}  |  ep len: {mean_l:.0f}")
