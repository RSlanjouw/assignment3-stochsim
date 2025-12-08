import simpy
import numpy as np
import matplotlib.pyplot as plt

def barage_queue_simulation(n_water=10000, n_queues=1, arrival_rate=1.0, queue_speed=[1.0, 0.25], start_level=1000, arrival_rate_func=None):
    env = simpy.Environment()
    barage_lake = simpy.Resource(env, capacity=n_queues)
    waiting_times = []
    service_times = []
    interval = []
    arrival_rate_history = [arrival_rate]
    queue_lengths = []
    queue = [0]
    last_water = env.event()

    def queue_time(queue_speed):
        mu, sigma = queue_speed
        service_time = -1
        while service_time < 0:
            service_time = np.random.normal(mu, sigma)
        return service_time

    def check_passenger(env, barage_lake, queue_speed):
        queue_lengths.append(queue[0])
        queue[0] += 1
        arrival_time = env.now
        with barage_lake.request() as request:
            yield request
            wait = env.now - arrival_time
            waiting_times.append(wait)
            service = queue_time(queue_speed)
            service_times.append(service)
            yield env.timeout(service)
            queue[0] -= 1

    current_arrival_rate = [arrival_rate]

    def passenger_generator(env, barage_lake, n_water, queue_speed):
        for i in range(n_water):
            arrival = np.random.exponential(1.0 / current_arrival_rate[0])
            interval.append(arrival)
            if i == n_water - 1:
                last_water.succeed()
            yield env.timeout(arrival)
            passenger = check_passenger(env, barage_lake, queue_speed)
            env.process(passenger)

    def arrival_rate_changer(env, current_arrival_rate):
        t = 0
        while True:
            yield env.timeout(10)
            t += 10
            if arrival_rate_func is not None:
                current_arrival_rate[0] = arrival_rate_func(current_arrival_rate[0], t)
            arrival_rate_history.append(current_arrival_rate[0])

    def generate_initial_passengers(env, barage_lake, n_initial, queue_speed):
        for _ in range(n_initial):
            passenger = check_passenger(env, barage_lake, queue_speed)
            env.process(passenger)
            # interval.append(0.001)
            yield env.timeout(0.001)

    env.process(generate_initial_passengers(env, barage_lake, start_level, queue_speed))
    env.process(passenger_generator(env, barage_lake, n_water, queue_speed))
    env.process(arrival_rate_changer(env, current_arrival_rate))
    env.run(until=last_water)

    return waiting_times, service_times, interval, arrival_rate_history, queue_lengths
