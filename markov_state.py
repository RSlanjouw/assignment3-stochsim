from modules.assignment3 import barage_queue_simulation
import numpy as np
import matplotlib.pyplot as plt


SEED = 1001
np.random.seed(SEED)

# Parameters
n_water = 100000
n_queues = 0.999
arrival_rate = 1.0
queue_speed = [1.0, 0.25]
start_level = 1000


print("Arrival rate")

# Hidden Markov Model as a functional generator (now with 3 states: dry, rain, storm)
def make_hmm_arrival_rate_func(
    P=np.array([
        [0.4, 0.5, 0.1],  # dry -> dry, rain, storm
        [0.1, 0.8, 0.1],  # rain -> dry, rain, storm
        [0.1, 0.7, 0.2]   # storm -> dry, rain, storm
    ]),
    rates=(0.5, 0.85, 3.0),  # arrival rates for dry, rain, storm
    start_state=0,
    step_interval=10
):
    state = {'val': start_state}
    def hmm_func(current_rate, t):
        if t % step_interval == 0:
            state['val'] = np.random.choice([0, 1, 2], p=P[state['val']])
        return rates[state['val']]
    return hmm_func



def calculate_stationary_distribution(P):
    w, v = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(w - 1.0))
    pi = np.real(v[:, idx])
    pi = pi / pi.sum()
    return pi

# Example: HMM with 3 states
hmm_arrival_rate_func = make_hmm_arrival_rate_func()


# Run simulation with HMM arrival rate
waiting_times, service_times, interval, arrival_rate_history, queue_lengths = barage_queue_simulation(
    n_water=n_water,
    n_queues=n_queues,
    arrival_rate=arrival_rate,
    queue_speed=queue_speed,
    start_level=start_level,
    arrival_rate_func=hmm_arrival_rate_func
)


# Plotting
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Arrival rate over time
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

# Plot 3: Waiting time distribution
axs[1, 0].hist(waiting_times, bins=50)
axs[1, 0].set_title('Waiting Time Distribution')
axs[1, 0].set_xlabel('Waiting Time')
axs[1, 0].set_ylabel('Count')
axs[1, 0].set_ylim(bottom=1)

# Plot 4: Interval time distribution
axs[1, 1].hist(interval, bins=50)
axs[1, 1].set_title('Interval Time Distribution')
axs[1, 1].set_xlabel('Interval Time')
axs[1, 1].set_ylabel('Count')
# make 0 y

plt.tight_layout()
plt.savefig('images/simulation_plots_markov.png')
# plt.show()




P = np.array([
        [0.4, 0.5, 0.1],  # dry -> dry, rain, storm
        [0.1, 0.8, 0.1],  # rain -> dry, rain, storm
        [0.1, 0.7, 0.2]   # storm -> dry, rain, storm
    ])
rates = (0.5, 0.80, 2.9)
h = calculate_stationary_distribution(P) @ np.array(rates)
print(f"Expected arrival rate (stationary distribution): {h}")