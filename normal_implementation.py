from modules.assignment3 import barage_queue_simulation
import numpy as np
import matplotlib.pyplot as plt

seed = 32
np.random.seed(seed)
# Parameters
n_water = 100000
n_queues = 1
arrival_rate = 0.99
queue_speed = [1.0, 0.25]
start_level = 1000

# Simulatie draaien
waiting_times, service_times, interval, arrival_rate_history, queue_lengths = barage_queue_simulation(
    n_water=n_water,
    n_queues=n_queues,
    arrival_rate=arrival_rate,
    queue_speed=queue_speed,
    start_level=start_level,
    arrival_rate_func=None,
    pairing=True
)

# Plotten
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Arrival rate over tijd
axs[0, 0].plot(arrival_rate_history)
axs[0, 0].set_title('Arrival Rate Over Time')
axs[0, 0].set_xlabel('Time')
axs[0, 0].set_ylabel('Arrival Rate')

# Plot 2: Queue length over time
service_times_cum = np.cumsum(service_times)
min_len = min(len(service_times_cum), len(queue_lengths[start_level:]))
axs[0, 1].plot(service_times_cum[:min_len], queue_lengths[start_level:start_level+min_len])
axs[0, 1].set_title('Queue Length Over Time')
axs[0, 1].set_xlabel('Time')
axs[0, 1].set_ylabel('Queue Length')

# Plot 3: Waiting time Distribution
axs[1, 0].hist(waiting_times, bins=50)
axs[1, 0].set_title('Waiting Time Distribution')
axs[1, 0].set_xlabel('Waiting Time')
axs[1, 0].set_ylabel('Frequency')

# Plot 4: Interval time Distribution
axs[1, 1].hist(interval, bins=50)
axs[1, 1].set_title('Interval Time Distribution')
axs[1, 1].set_xlabel('Interval Time')
axs[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('images/simulation_plots_normal.png')
# plt.show()