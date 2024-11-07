import numpy as np
from tqdm import trange

def sphere_function(x):
    return np.sum(x**2)

def evolution_strategy(strategy_type="1+1", sigma=0.01, max_iters=10000000, target=0.005, n_runs=10):
    results = []
    N = 10
    G = 100

    a = 1.5
    np.random.seed(42)
    tau = 1 / np.sqrt(2 * np.sqrt(N))
    tau_prime = 1 / np.sqrt(2 * N)
    print(f"tau: {tau}, tau_prime: {tau_prime}")
    epsilon = 1e-6  # Small lower bound for step sizes to avoid collapse to zero
    upper_bound = sigma*10
    for _ in trange(n_runs):
        x = np.ones(N)  # Start at (1, 1, ..., 1)
        cur_sigma = sigma
        iterations = 0
        success = 0
        while iterations < max_iters:
            # Generate offspring with Gaussian mutation
            offspring = x + np.random.normal(0, cur_sigma, N)

            # sigmas = sigmas * np.exp(tau_prime * np.random.randn() + tau * np.random.randn(N))
            # sigmas = np.clip(sigmas, epsilon, upper_bound)
            # sigmas = np.maximum(sigmas, epsilon)
            # print(sigmas)
            # Individual mutation
            # offspring = x + sigmas * np.random.randn(N)
            
            # Evaluate fitness
            f_parent = sphere_function(x)
            f_offspring = sphere_function(offspring)
            
            # Selection based on the strategy
            if strategy_type == "1,1":
                x = offspring  # Always replace parent
                success += 1 if f_offspring < f_parent else 0
                
            elif strategy_type == "1+1":
                if f_offspring < f_parent:
                    x = offspring  # Replace only if offspring is better
                    success += 1
            # 1/5th rule
            if (iterations + 1) % G == 0:
                success_rate = success / G
                if success_rate > 0.2:
                    cur_sigma *= a
                else:
                    cur_sigma /= a
                cur_sigma = np.clip(cur_sigma, epsilon, upper_bound)
                # print(f"Success rate: {success_rate}, Sigma: {cur_sigma}")
                success = 0
            # Check termination criteria
            if sphere_function(x) <= target:
                break

            iterations += 1
        if iterations == max_iters:
            print("Max iterations reached")
        # Record the number of iterations taken for this run
        results.append(iterations)
    
    return results

# Run experiments
sigmas = [0.01, 0.1, 1.0]
for sigma in sigmas:
    print(f"Sigma: {sigma}")
    # print("(1 , 1)-ES Results:", evolution_strategy("1,1", sigma=sigma))
    print("(1 + 1)-ES Results:", evolution_strategy("1+1", sigma=sigma))
