from modules.assignment3 import barage_queue_simulation
import numpy as np
import math
import matplotlib.pyplot as plt


def calculate_cost(queue_lengths, target_level=1000, capacity=2000):
    data = np.array(queue_lengths)[1000:] # discard warmup
    if len(data) == 0: return 1e9

    # penalize floods quadratically * 5, to penalize potential for catastrophes
    excess_water = np.maximum(0, data - capacity)
    flood_cost = np.sum(excess_water**2) * 5 

    # every point under min level also quadratically penalized to prevent economic loss 
    min_level = 200 
    water_deficit = np.maximum(0, min_level - data)
    drought_cost = np.sum(water_deficit**2)

    # target deviation penalty to act as a tiebreaker if no floods and no droughts, soft penalty and linear
    target_cost = np.mean(np.abs(data - target_level)) * 0.5

    return flood_cost + drought_cost + target_cost

def get_average_cost(n_runs, n_water, queue_speed):
    costs = []
    for _ in range(n_runs):
        # use pairing for variance reduction
        _, _, _, _, q_lengths = barage_queue_simulation(
            n_water=n_water, 
            queue_speed=queue_speed, 
            pairing=True
        )
        costs.append(calculate_cost(q_lengths))
    return np.mean(costs)

def simulated_annealing(n_iterations=500, initial_temp=1000, runs_per_step=3):
    b_param = 2
    
    # make sure cooling curve starts close to initial_temp
    a_param = initial_temp * math.log(b_param)
    current_speed_mean = 1.0 
    current_cost = get_average_cost(runs_per_step, 5000, [current_speed_mean, 0.25])
    
    best_speed = current_speed_mean
    best_cost = current_cost
    
    temp = initial_temp
    history_cost = []
    history_speed = []
    history_temp = []

    print(f"Start SA. Initial Cost: {current_cost:.2f}, Speed: {current_speed_mean:.3f}")

    for i in range(1, n_iterations + 1):
        # cool temp
        temp = a_param / math.log(i + b_param)
        
        # propose new speed
        step_size = 0.5 * (temp / initial_temp) 
        step_size = max(step_size, 0.05) 
        
        # use normal for better fine-tuning compared to jumping around
        proposal = current_speed_mean + np.random.normal(0, step_size)
        proposal = max(0.1, proposal) 
        
        # run multiple simulations
        new_cost = get_average_cost(runs_per_step, 5000, [proposal, 0.25])
        
        
        cost_diff = new_cost - current_cost
        
        # accept or reject
        if cost_diff < 0 or np.random.rand() < math.exp(-cost_diff / temp):
            current_speed_mean = proposal
            current_cost = new_cost
            
            # update
            if current_cost < best_cost:
                best_cost = current_cost
                best_speed = current_speed_mean
                print(f"  new record! Cost: {best_cost:.2f} (Speed: {best_speed:.3f})")

        history_cost.append(current_cost)
        history_speed.append(current_speed_mean)
        history_temp.append(temp)
        
        if i % 20 == 0:
            print(f"Iter {i}: Cost={current_cost:.2f}, Speed={current_speed_mean:.3f}, Temp={temp:.2f}")

    return best_speed, best_cost, history_cost, history_speed, history_temp


if __name__ == "__main__":
    best_param, min_cost, costs, params, temps = simulated_annealing(n_iterations=2000, runs_per_step=3)
    
    print(f"\noptimilization finished!")
    print(f"Best 'Mean Service Time' found: {best_param:.4f}")
    print(f"Min cost: {min_cost:.2f}")

    # Plot
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(costs)
    plt.title("Cost Evolution (Lower is Better)")
    plt.xlabel("Iteration")
    plt.ylabel("Average Cost")
    
    plt.subplot(1, 3, 2)
    plt.plot(params)
    plt.axhline(y=best_param, color='r', linestyle='--', label='Best Found')
    plt.title("Parameter Search (Outflow Speed)")
    plt.xlabel("Iteration")
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(temps, color='orange')
    plt.title("Logarithmic Cooling")
    plt.xlabel("Iteration")
    plt.ylabel("Temperature")
    
    plt.tight_layout()
    # plt.savefig('optimization_results.png')
    print("Plot opgeslagen als 'optimization_results.png'")
    plt.show()