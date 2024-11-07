import numpy as np
from tqdm import trange

def sphere_function(x):
    return np.sum(x**2)

def evolution_strategy(mu, lambda_, sigma, max_iters=10000000, target=0.005, n_runs=10):
    results = []
    N = 10
    np.random.seed(42)
    for _ in trange(n_runs):
        x = np.ones(N)  # Start at (1, 1, ..., 1)
        # ttt = tqdm.tqdm(range(max_iters))
        iterations = 0
        while iterations < max_iters:
            # Generate offspring with Gaussian mutation
            offspring = x + np.random.normal(0, sigma, N)
            
            # Evaluate fitness
            f_parent = sphere_function(x)
            f_offspring = sphere_function(offspring)
            
            # Selection
            if lambda_ == 1:  # (1,1)-ES
                x = offspring  # Offspring replaces parent
            else:  # (1+1)-ES
                if f_offspring < f_parent:
                    x = offspring  # Offspring replaces parent if better
            
            # Check termination criteria
            if (sphere_function(x)) <= target:
                break

            iterations += 1
        if iterations == max_iters:
            print("Max iterations reached")
        # Record the number of iterations taken for this run
        # print(sphere_function(x))
        results.append(iterations)
    
    return results

# Run experiments
sigmas = [0.01, 0.1, 1.0]
for sigma in sigmas:
    print(f"Sigma: {sigma}")
    # print("(1 , 1)-ES Results:", evolution_strategy(mu=1, lambda_=1, sigma=sigma))
    print("(1 + 1)-ES Results:", evolution_strategy(mu=1, lambda_=2, sigma=sigma))
