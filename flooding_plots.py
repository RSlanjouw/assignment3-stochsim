import matplotlib.pyplot as plt
import numpy as np

def plot_min_max_boxplot(min_queues, max_queues, filename='images/flooding_minmax_boxplot_markov.png'):
    mean_max = np.mean(max_queues)
    std_max = np.std(max_queues)
    ci95_max = 1.96 * std_max / np.sqrt(len(max_queues))
    mean_min = np.mean(min_queues)
    std_min = np.std(min_queues)
    ci95_min = 1.96 * std_min / np.sqrt(len(min_queues))

    plt.figure(figsize=(12, 6))
    plt.boxplot([min_queues, max_queues], labels=['Min', 'Max'])
    plt.title('Min/Max Queue Lengths with Confidence Interval')
    plt.ylabel('Queue Length')
    plt.grid(True, axis='y')
    plt.text(1, mean_min, f"Mean: {mean_min:.1f}\nCI95: ±{ci95_min:.1f}", ha='center', va='bottom', color='blue')
    plt.text(2, mean_max, f"Mean: {mean_max:.1f}\nCI95: ±{ci95_max:.1f}", ha='center', va='bottom', color='red')
    plt.savefig(filename)
    plt.close()
    print(f"Saved min/max boxplot as '{filename}'.")

def plot_max_distribution(max_queues, filename='images/flooding_max_distribution_markov.png'):
    plt.figure(figsize=(12, 6))
    bins = np.arange(int(np.min(max_queues)), int(np.max(max_queues)) + 50, 50)
    counts, edges = np.histogram(max_queues, bins=bins)
    plt.bar(edges[:-1], counts, width=45, align='edge', alpha=0.6, label='Aantal keer max waarde')
    cum_counts = np.array([np.sum(max_queues >= b) for b in bins])
    plt.plot(bins, cum_counts, marker='o', color='red', label='Aantal runs met max ≥ waarde')
    plt.xlabel('Maximale Queue Lengte')
    plt.ylabel('Aantal runs')
    plt.title('Hoe vaak wordt een bepaalde max waarde bereikt?')
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig(filename)
    plt.close()
    print(f"Saved max value distribution plot as '{filename}'.")

def plot_min_distribution(min_queues, filename='images/flooding_min_distribution_markov.png'):
    plt.figure(figsize=(12, 6))
    bins = np.arange(int(np.min(min_queues)), int(np.max(min_queues)) + 50, 50)
    counts, edges = np.histogram(min_queues, bins=bins)
    plt.bar(edges[:-1], counts, width=45, align='edge', alpha=0.6, label='Aantal keer min waarde')
    cum_counts = np.array([np.sum(min_queues <= b) for b in bins])
    plt.plot(bins, cum_counts, marker='o', color='blue', label='Aantal runs met min ≤ waarde')
    plt.xlabel('Minimale Queue Lengte')
    plt.ylabel('Aantal runs')
    plt.title('Hoe vaak wordt een bepaalde min waarde bereikt?')
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig(filename)
    plt.close()
    print(f"Saved min value distribution plot as '{filename}'.")

def main():
    max_queues = np.load("data/max_queues_markov.npy")
    min_queues = np.load("data/min_queues_markov.npy")
    plot_min_max_boxplot(min_queues, max_queues)
    plot_max_distribution(max_queues)
    plot_min_distribution(min_queues)

if __name__ == "__main__":
    main()