import numpy as np
from scipy import stats
from modules.assignment3 import barage_queue_simulation


def test_stratified_sampling ():
    n_runs = 40        
    arrival_rate = 1.0 
    
    levels_standard = []
    levels_stratified = []
    
    for seed in range(n_runs):
        np.random.seed(seed) 
        
        # standard monte carlo
        _, _, _, _, waterlevel_normal = barage_queue_simulation(n_water=5000, arrival_rate=1.0, start_level=1000, pairing=False)
        levels_standard.append(np.mean(waterlevel_normal))
        
        # stratified sampling
        _, _, _, _, waterlevel_stratified = barage_queue_simulation(n_water=5000, arrival_rate=arrival_rate, start_level=1000, pairing=False, stratified=True)
        levels_stratified.append(np.mean(waterlevel_stratified))
        
    var_montecarlo = np.var(levels_standard, ddof=1)
    var_stratified = np.var(levels_stratified, ddof=1)
    # f-test for unequal variances
    levene_stat, p_value_var = stats.levene(levels_standard, levels_stratified)
    
    print(f"Variance of Standard Monte Carlo: {var_montecarlo:.6f}")
    print(f"Variance of Stratified Sampling: {var_stratified:.6f}")
    print(f"Levene Statistic (F-test for unequal variances): {levene_stat:.4f}, p-value: {p_value_var:.4e}")


if __name__ == "__main__":
    test_stratified_sampling()