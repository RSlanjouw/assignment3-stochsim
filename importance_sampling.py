import numpy as np
from modules.assignment3 import barage_queue_simulation

def importance_sampling():
    n_runs = 1000
    normal_lambda = 0.85  # f(x): normal weather
    storm_lambda = 0.95  # g(x): stormy weather 
    start_level = 800
    flood_threshold = 1100  # used lower value 
    n_water = 500
    
    # standard monte carlo
    mc_is_flood = []

    for _ in range(n_runs):
        _, _, intervals, _, queue_lengths = barage_queue_simulation(n_water, normal_lambda, start_level)
        mc_is_flood.append(1.0 if np.any(np.array(queue_lengths) > flood_threshold) else 0.0)
    
    prob_mc = np.mean(mc_is_flood)
    var_mc = np.var(mc_is_flood, ddof=1) / n_runs
    flood_count_mc = sum(mc_is_flood)

    # importance sampling
    flood_weighted = []
    importance_is_flood = []

    for _ in range(n_runs):
        _, _, intervals, _, queue_lengths = barage_queue_simulation(n_water, storm_lambda, start_level)
        is_flood = 1.0 if np.any(np.array(queue_lengths) > flood_threshold) else 0.0
        importance_is_flood.append(is_flood)
        intervals = np.array(intervals)
        log_weight = np.sum(np.log(normal_lambda) - normal_lambda * intervals -
                    (np.log(storm_lambda) - storm_lambda * intervals))
        weighted = np.exp(log_weight) * is_flood
        flood_weighted.append(weighted)

    prob_is = np.mean(flood_weighted)
    var_is = np.var(flood_weighted, ddof=1) / n_runs
    flood_count_is = sum(importance_is_flood)

    print(f"Standard Monte Carlo:")
    print(f"Observed floods in Standard Monte Carlo: {flood_count_mc}/{n_runs} ")
    print(f"Estimated probability of flooding: {prob_mc:.4e}")
    print(f"Variance of estimate: {var_mc:.4e}", "\n")
    print(f"Importance Sampling:")
    print(f"Observed floods in importance sampling: {flood_count_is}/{n_runs}")
    print(f"Estimated probability of flooding: {prob_is:.4e}")
    print(f"Variance of estimate: {var_is:.4e}", "\n")
    print(f"Variance reduction ratio (Monte Carlo / Importance Sampling): {var_mc / var_is}")


if __name__ == "__main__":
    importance_sampling()
