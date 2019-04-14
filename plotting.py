import numpy as np
import matplotlib.pyplot as plt

rewards = []
frames = []
with open("logging.txt", "r") as file:
    lines = file.readlines()
    for line in lines:
        reward = float(line.split()[3])
        frame = int(line.split()[1].strip("frames:"))
        rewards.append(reward)
        frames.append(frame)


plt.plot(frames, rewards)
plt.ylabel("Rewards")
plt.show()