import numpy as np
import matplotlib.pyplot as plt

tau = 30  # radius of moscow MKAD
k = np.log(2) / tau  # decay rate
d = np.linspace(0, 1000, 1000)  # distance from 0 to 1000 km

# Decay function
f_d = np.exp((-k * d) / 2)


plt.figure(figsize=(10, 6))
plt.plot(d, f_d, label=r"$f_d = e^{-kd/2}$", color="royalblue")
plt.title("Distance Favorability Constant over Distance", fontsize=12)
plt.xlabel("Distance (km)", fontsize=10)
plt.ylabel(r"$f_d$", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
