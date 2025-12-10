import random
import math
import matplotlib.pyplot as plt
from modules.assignment3 import barage_queue_simulation
from markov_state import make_hmm_arrival_rate_func
from sde_levy import create_levy_arrival_rate_func
# ===============================
# 1. Probleemdefinitie
# ===============================
import numpy as np
np.random.seed(299421222)
# Het landschap (hoogtes)
hmm = create_levy_arrival_rate_func(
    sigma=0.5,       # verhoogd voor meer diffusie (schommeling)
    jump_lambda=0.1, # verhoogd voor meer sprongen
    jump_sigma=1.0,  # verhoogd voor grotere sprongen
    theta=0.3        # drift blijft gelijk
)

_, _, _, _, queue = barage_queue_simulation(
    n_water=100000,  # langer voor meer schommelingen
    n_queues=1,
    arrival_rate=1.0,
    queue_speed=[1.0, 0.25],
    start_level=500,
    arrival_rate_func=hmm
)

array = queue[1000:]
n = len(array)

# ===============================
# 2. Simulated Annealing parameters
# ===============================
n_runs = 5  # aantal runs om de beste te vinden
T_start = 1000.0        # starttemperatuur
T_min = 0.001          # stopt temperatuur
cooling_rate = 0.995   # nog langzamere afkoeling voor betere escape
max_iterations = 3000

# ===============================
# 3. Initialisatie
# ===============================

current_index = random.randint(0, n - 1)
current_value = array[current_index]

best_index = current_index
best_value = current_value

T = T_start

# Logging voor plots
index_history = [current_index]
value_history = [current_value]
temperature_history = [T]

# ===============================
# 4. Simulated Annealing loop
# ===============================

iteration = 0

while T > T_min and iteration < max_iterations:
    iteration += 1

    # Kies buur (veel grotere stap voor de grote array)
    step = random.randint(-5000, 5000)  # grote sprongen voor 100k array
    new_index = current_index + step

    # Reflecteer aan grenzen om bouncing toe te staan
    if new_index < 0:
        new_index = -new_index
    elif new_index >= n:
        new_index = 2 * n - new_index - 1

    new_value = array[new_index]
    delta = new_value - current_value

    # Acceptatiecriterium
    if delta > 0:
        accept = True
    else:
        probability = math.exp(delta / T)
        accept = random.random() < probability

    if accept:
        current_index = new_index
        current_value = new_value

        if current_value > best_value:
            best_index = current_index
            best_value = current_value

    # Log data
    index_history.append(current_index)
    value_history.append(current_value)

    # Koel af
    T *= cooling_rate
    temperature_history.append(T)

# ===============================
# 5. Resultaten printen
# ===============================

print("=== RESULTAAT ===")
print(f"Beste index: {best_index}")
print(f"Beste waarde: {best_value}")

# ===============================
# 6. Plotten
# ===============================

# ===============================
# 6. Plotten
# ===============================
plt.figure(figsize=(20, 5))

# Plot 1: Array landschap
plt.subplot(1, 4, 1)
plt.plot(array)
plt.scatter(best_index, best_value, color='red')
plt.title("Array Landscape")
plt.xlabel("Index")
plt.ylabel("Value")

# Plot 2: Onderzochte index
plt.subplot(1, 4, 2)
plt.plot(index_history)
plt.title("Explored Index")
plt.xlabel("Iteration")
plt.ylabel("Index")

# Plot 3: Waarde
plt.subplot(1, 4, 3)
plt.plot(value_history)
plt.title("Value Evolution")
plt.xlabel("Iteration")
plt.ylabel("Value")

# Plot 4: Temperatuur
plt.subplot(1, 4, 4)
plt.plot(temperature_history)
plt.title("Temperature Evolution")
plt.xlabel("Iteration")
plt.ylabel("Temperature")

plt.tight_layout()
plt.show()