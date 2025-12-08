### WORK IN PROGRESSSSSS

from modules.assignment3 import barage_queue_simulation
import numpy as np  
from scipy import stats

# anhealing is the pair varable. 


def test_anhealing():
    # Hypotheses test: Is er verschil in gemiddelde queue size en variance tussen pairing=False en pairing=True?
    # Voor verschil in mean: H0: mean_false == mean_true
    # Voor variance: H0: var_false == var_true (maar verwacht var_true < var_false voor variance reduction)
    # Er is geen verschil in mean omdat dezelfde parameters gebruikt worden, maar variance kan verschillen door antithetic variates.
    
    n_runs = 30
    means_false = []
    means_true = []
    vars_false = []
    vars_true = []
    
    for seed in range(n_runs):
        np.random.seed(seed)
        
        # Draai simulatie zonder pairing
        _, _, _, _, queue_lengths_false = barage_queue_simulation(n_water=10000, pairing=False)
        mean_false = np.mean(queue_lengths_false)
        var_false = np.var(queue_lengths_false, ddof=1)
        means_false.append(mean_false)
        vars_false.append(var_false)
        
        # Draai simulatie met pairing (anhealing)
        _, _, _, _, queue_lengths_true = barage_queue_simulation(n_water=10000, pairing=True)
        mean_true = np.mean(queue_lengths_true)
        var_true = np.var(queue_lengths_true, ddof=1)
        means_true.append(mean_true)
        vars_true.append(var_true)
    
    # Bereken overall stats
    avg_mean_false = np.mean(means_false)
    avg_mean_true = np.mean(means_true)
    avg_var_false = np.mean(vars_false)
    avg_var_true = np.mean(vars_true)
    
    # Variance van de sample means (voor variance reduction effect)
    var_of_means_false = np.var(means_false, ddof=1)
    var_of_means_true = np.var(means_true, ddof=1)
    
    print(f"Gemiddelde van means queue size zonder pairing: {avg_mean_false:.4f}")
    print(f"Gemiddelde van means queue size met pairing: {avg_mean_true:.4f}")
    print(f"Gemiddelde variance queue size zonder pairing: {avg_var_false:.4f}")
    print(f"Gemiddelde variance queue size met pairing: {avg_var_true:.4f}")
    print(f"Variance van sample means zonder pairing: {var_of_means_false:.6f}")
    print(f"Variance van sample means met pairing: {var_of_means_true:.6f}")
    
    if var_of_means_true < var_of_means_false:
        reduction_factor = var_of_means_false / var_of_means_true
        print(f"Variance reduction factor: {reduction_factor:.2f} (pairing vermindert variance van estimator)")
    else:
        print("Geen variance reduction waargenomen.")
    
    # Covariantie tussen means_false en means_true
    cov = np.cov(means_false, means_true)[0,1]
    print(f"Covariantie tussen means_false en means_true: {cov:.4f}")
    
    # T-test voor verschil in gemiddelde means
    t_stat_mean, p_value_mean = stats.ttest_ind(means_false, means_true)
    print(f"T-test voor means: t={t_stat_mean:.4f}, p={p_value_mean:.4f}")
    if p_value_mean < 0.05:
        print("  Significant verschil in gemiddelde queue size.")
    else:
        print("  Geen significant verschil in gemiddelde queue size (verwacht, want dezelfde parameters).")
    
    # Levene's test voor verschil in variances (van de interne variances)
    levene_stat, p_value_var = stats.levene(vars_false, vars_true)
    print(f"Levene's test voor variances: W={levene_stat:.4f}, p={p_value_var:.4f}")
    if p_value_var < 0.05:
        print("  Significant verschil in variances.")
        if avg_var_true < avg_var_false:
            print("  Pairing vermindert interne variance.")
        else:
            print("  Pairing verhoogt interne variance (onverwacht).")
    else:
        print("  Geen significant verschil in variances.")


if __name__ == "__main__":
    test_anhealing()
