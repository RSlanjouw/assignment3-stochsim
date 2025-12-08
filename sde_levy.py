
from modules.assignment3 import barage_queue_simulation
import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
n_water = 100000
n_queues = 1
arrival_rate = 1.0
queue_speed = [1.0, 0.25]
start_level = 1000

def create_levy_arrival_rate_func(
    sigma=0.1,
    jump_lambda=0.08,
    jump_mu=0.0,
    jump_sigma=0.15,
    theta=0.4,
    rate_min=0.0001,
    rate_max=5
):
    """
    Returns a function that updates the arrival rate using a mean-reverting Lévy SDE.
    SDE: dX = θ(1-X)dt + σdW + jumps
    """
    rate_state = [1.0]
    def arrival_rate_func(_, t):
        dt = 10
        dt_scaled = dt / 10.0
        drift = theta * (1.0 - rate_state[0]) * dt_scaled
        diffusion = sigma * np.random.normal(0, np.sqrt(dt_scaled))
        n_jumps = np.random.poisson(jump_lambda * dt_scaled)
        jump_total = np.sum(np.random.normal(jump_mu, jump_sigma, n_jumps)) if n_jumps > 0 else 0.0
        rate_state[0] += drift + diffusion + jump_total
        rate_state[0] = np.clip(rate_state[0], rate_min, rate_max)
        return rate_state[0]
    return arrival_rate_func

# SDE parameters for simulation
arrival_rate_func = create_levy_arrival_rate_func(
    sigma=0.1,
    jump_lambda=0.01,
    jump_sigma=0.2,
    theta=0.3
)

# Run simulation
waiting_times, service_times, interval, arrival_rate_history, queue_lengths = barage_queue_simulation(
    n_water=n_water,
    n_queues=n_queues,
    arrival_rate=arrival_rate,
    queue_speed=queue_speed,
    start_level=start_level,
    arrival_rate_func=arrival_rate_func
)

# Plot results
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs[0, 0].plot(arrival_rate_history)
axs[0, 0].set_title('Arrival Rate Over Time')
axs[0, 0].set_xlabel('Time')
axs[0, 0].set_ylabel('Arrival Rate')

service_times_cum = np.cumsum(service_times)
min_len = min(len(service_times_cum), len(queue_lengths[start_level:]))
axs[0, 1].plot(service_times_cum[:min_len], queue_lengths[start_level:start_level+min_len])
axs[0, 1].set_title('Queue Length Over Time')
axs[0, 1].set_xlabel('Time')
axs[0, 1].set_ylabel('Queue Length')

axs[1, 0].hist(waiting_times, bins=50)
axs[1, 0].set_title('Waiting Time Distribution')
axs[1, 0].set_xlabel('Waiting Time')
axs[1, 0].set_ylabel('Count')

axs[1, 1].hist(interval, bins=50)
axs[1, 1].set_title('Interval Time Distribution')
axs[1, 1].set_xlabel('Interval Time')
axs[1, 1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('images/simulation_plots_sde.png')
# plt.show()