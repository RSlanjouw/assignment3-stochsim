from assignment3 import barage_queue_simulation
import numpy as np
import matplotlib.pyplot as plt
# Parameters
n_water = 100000
n_queues = 1
arrival_rate = 0.9
queue_speed = [1.0, 0.25]
start_level = 1000


# SDE with Lévy jumps as arrival rate function
def levy_sde_arrival_rate_func(mu=1.0, sigma=0.1, jump_lambda=0.05, jump_mu=0.5, jump_sigma=0.2):
    rate = [mu]
    t_last = [0]
    def func(current_rate, t):
        dt = t - t_last[0]
        t_last[0] = t
        # Brownian motion part
        dW = np.random.normal(0, np.sqrt(dt))
        # Lévy jump part
        n_jumps = np.random.poisson(jump_lambda * dt)
        jumps = np.sum(np.random.normal(jump_mu, jump_sigma, n_jumps)) if n_jumps > 0 else 0.0
        rate[0] = max(0.01, rate[0] + mu * dt + sigma * dW + jumps)
        return rate[0]
    return func

# Example: use SDE with Lévy jumps for arrival rate
arrival_rate_func = levy_sde_arrival_rate_func(mu=0.0, sigma=0.05, jump_lambda=0.02, jump_mu=0.5, jump_sigma=0.1)

# Run simulation
waiting_times, service_times, interval, arrival_rate_history, queue_lengths = barage_queue_simulation(
    n_water=n_water,
    n_queues=n_queues,
    arrival_rate=arrival_rate,
    queue_speed=queue_speed,
    start_level=start_level,
    arrival_rate_func=arrival_rate_func
)

# Plotten
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Arrival rate over tijd
axs[0, 0].plot(arrival_rate_history)
axs[0, 0].set_title('Arrival Rate Over Tijd')
axs[0, 0].set_xlabel('Tijd')
axs[0, 0].set_ylabel('Arrival Rate')

# Plot 2: Queue lengte over tijd
service_times_cum = np.cumsum(service_times)
min_len = min(len(service_times_cum), len(queue_lengths[start_level:]))
axs[0, 1].plot(service_times_cum[:min_len], queue_lengths[start_level:start_level+min_len])
axs[0, 1].set_title('Queue Lengte Over Tijd')
axs[0, 1].set_xlabel('Tijd')
axs[0, 1].set_ylabel('Queue Lengte')

# Plot 3: Waiting time distributie
axs[1, 0].hist(waiting_times, bins=50)
axs[1, 0].set_title('Waiting Time Distributie')
axs[1, 0].set_xlabel('Waiting Time')
axs[1, 0].set_ylabel('Aantal')

# Plot 4: Interval time distributie
axs[1, 1].hist(interval, bins=50)
axs[1, 1].set_title('Interval Time Distributie')
axs[1, 1].set_xlabel('Interval Time')
axs[1, 1].set_ylabel('Aantal')

plt.tight_layout()
plt.savefig('simulation_plots_sde.png')
# plt.show()