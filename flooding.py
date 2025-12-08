from modules.assignment3 import barage_queue_simulation
from sde_levy import create_levy_arrival_rate_func
from markov_state import make_hmm_arrival_rate_func
import numpy as np


def run_multiple_simulations(n_runs=1000, seed_start=0):
    n_water = 100000
    n_queues = 1
    arrival_rate = 1.0
    queue_speed = [1.0, 0.25]
    start_level = 1000
    min_queues = []
    max_queues = []
    for i in range(n_runs):
        np.random.seed(seed_start + i)
        hmm_arrival_rate_func = make_hmm_arrival_rate_func()

        arrival_rate_func = hmm_arrival_rate_func


        _, _, _, _, queue_lengths = barage_queue_simulation(
            n_water=n_water,
            n_queues=n_queues,
            arrival_rate=arrival_rate,
            queue_speed=queue_speed,
            start_level=start_level,
            arrival_rate_func=arrival_rate_func,
        )
        # Only consider after startup
        queue_after_startup = queue_lengths[start_level:]
        if len(queue_after_startup) > 0:
            min_queues.append(np.min(queue_after_startup))
            max_queues.append(np.max(queue_after_startup))
        if (i % 10) == 0:
            print(f"Completed {i+1}/{n_runs} simulations.")
        print(f"Completed {i+1}/{n_runs} simulations.")

    # Save results in data folder
    np.save("data/min_queues_markov.npy", np.array(min_queues))
    np.save("data/max_queues_markov.npy", np.array(max_queues))
    print(f"Saved min_queues and max_queues to data folder.")
    print(f"Over {n_runs} runs:")
    print(f"Minimum queue length after startup: {np.min(min_queues)}")
    print(f"Maximum queue length after startup: {np.max(max_queues)}")


if __name__ == "__main__":
    run_multiple_simulations()
