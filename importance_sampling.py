import numpy as np
from modules.assignment3 import barage_queue_simulation

def importance_sampling():
    '''
    Lecture: Variance Reduction Techniques
    Slides: 49 - 56
    '''
    n_runs = 1000
    normal_lambda = 1.0  # f(x): normal weather
    storm_lambda = 2.0  # g(x): stormy weather 
    flood_threshold = 1300  # used lower value 
    
    flood_weights = []
    flood_count = 0
    
    print(f"Running {n_runs} Importance Sampling simulations...")

    for _ in range(n_runs):
        _, _, _, _, waterlevel_normal = barage_queue_simulation(n_water=5000, arrival_rate=1.0, start_level=1000, pairing=False)
        levels_standard.append(np.mean(waterlevel_normal))

        # 1. sample from g(x): run simulation with storm_lambda to generate extreme (stormy) weather conditions
        _, _, intervals, _, queue_lengths = barage_queue_simulation(n_water=600, arrival_rate=storm_lambda, start_level=1000, pairing=False)
        
        # 2. h(x): 1 if flood (0 if no flood)
        if np.max(queue_lengths) > flood_threshold:  
            flood_count += 1   
            # estimator = (h(x) * f(x)) / g(x) 
            log_prob_normal = np.sum(np.log(normal_lambda) - normal_lambda * np.array(intervals)) 
            log_prob_storm = np.sum(np.log(storm_lambda) - storm_lambda * np.array(intervals))
            weight = np.exp(log_prob_normal - log_prob_storm)
            flood_weights.append(weight)
        else:
            # h(x) is 0 if no flood -> estimator = (0 * f(x)) = g(x) = 0
            flood_weights.append(0.0)

    prob_flood = np.mean(flood_weights) 
    var_montecarlo = prob_flood * (1.0 - prob_flood) / n_runs
    var_importance_sampling = np.var(np.array(flood_weights), ddof=1) / n_runs
    ratio = var_montecarlo / var_importance_sampling 

    print(f"Simulation used biased lambda to make storms more likely: {storm_lambda}")
    print(f"Observed floods in importance sampling simulation: {flood_count}/{n_runs}")
    print(f"Estimated probability of flooding: {prob_flood:.4e}")
    print(f"Variance of Standard Monte Carlo: {var_montecarlo}")
    print(f"Variance of Importance Sampling: {var_importance_sampling}")
    print(f"Ratio of Standard Monte Carlo to Importance Sampling: {ratio}")


if __name__ == "__main__":
    importance_sampling()

# computer is trying to work with values that are too tiny
# numerical instability if risk of event is too low?