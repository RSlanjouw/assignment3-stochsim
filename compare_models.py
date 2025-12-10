from modules.assignment3 import barage_queue_simulation
from sde_levy import create_levy_arrival_rate_func
from markov_state import make_hmm_arrival_rate_func
import numpy as np
from scipy import stats
import pandas as pd

def run_simulations ():
    n_runs = 40
    n_water = 5000
    start_level = 1000

    results_normal, results_markov, results_levy = [], [], []

    for i in range(n_runs):
        seed = 42 + i

        # normal model (constant water flow: lambda = 1.0)
        np.random.seed(seed)
        _, _, _, _, waterlevel_normal = barage_queue_simulation(n_water=n_water, arrival_rate=1.0, start_level=start_level, pairing=False)
        results_normal.append({
            'mean': np.mean(waterlevel_normal),
            'var': np.var(waterlevel_normal)
            })

        # markov model (HMM) (changing weather conditions: dry -> rain -> storm)
        np.random.seed(seed)
        hmm_water_flow = make_hmm_arrival_rate_func()
        _, _, _, _, waterlevel_markov = barage_queue_simulation(n_water=n_water, arrival_rate_func=hmm_water_flow, start_level=start_level, pairing=False)
        results_markov.append({
            'mean': np.mean(waterlevel_markov),
            'var': np.var(waterlevel_markov)
            })

        # levy model (sudden surges in weather conditions)
        np.random.seed(seed)
        levy_water_flow = create_levy_arrival_rate_func()
        _, _, _, _, waterlevel_levy = barage_queue_simulation(n_water=n_water, arrival_rate_func=levy_water_flow, start_level=start_level, pairing=False)
        results_levy.append({
            'mean': np.mean(waterlevel_levy),
            'var': np.var(waterlevel_levy)
            })
        
        normal, markov, levy = pd.DataFrame(results_normal), pd.DataFrame(results_markov), pd.DataFrame(results_levy)
    
    return normal, markov, levy


def compare_model_averages (normal, markov, levy):
    print("Average water levels:")
    print(f"1. Normal: {np.mean(normal['mean']):.2f}")
    print(f"2. Markov: {np.mean(markov['mean']):.2f}")
    print(f"3. Levy: {np.mean(levy['mean']):.2f}")

    # hypothesis test: normal vs. markov
    _, p1 = stats.ttest_ind(normal['mean'], markov['mean'], alternative='less', equal_var=False)
    print(f"Hypothesis test (Normal vs. Markov): p-value = {p1:.4f}")

    # hypothesis test: normal vs. levy 
    _, p2 = stats.ttest_ind(normal['mean'], levy['mean'], alternative='less', equal_var=False)
    print(f"Hypothesis test (Normal vs. Levy): p-value = {p2:.4f}", "\n")


def compare_model_variances (normal, markov, levy):
    print("Variance in water levels:")
    print(f"1. Normal: {np.mean(normal['var']):.4f}")
    print(f"2. Markov: {np.mean(markov['var']):.4f}")
    print(f"3. Levy: {np.mean(levy['var']):.4f}")

    # f-test: normal vs. markov
    _, p1 = stats.levene(normal['var'], markov['var'])
    print(f"Levene's test (Normal vs. Markov): p-value = {p1:.4f}")

    # f-test: normal vs. levy
    _, p2 = stats.levene(normal['var'], levy['var'])
    print(f"Levene's test (Normal vs. Levy): p-value = {p2:.4f}")

    # f-test: levy vs. markov
    _, p3 = stats.levene(levy['var'], markov['var'])
    print(f"Levene's test (Levy vs. Markov): p-value = {p3:.4f}")


if __name__ == "__main__":
    normal, markov, levy = run_simulations()
    compare_model_averages(normal, markov, levy)
    compare_model_variances(normal, markov, levy)
    