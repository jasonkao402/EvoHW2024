import operator
import random
from deap import base, creator, tools, gp, algorithms
from functools import partial
import numpy as np

# Step 1: Define the primitive set
pset = gp.PrimitiveSet("MAIN", 4)  # 4 inputs
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addEphemeralConstant("rand", partial(random.uniform, -1, 1))  # Random constant
pset.renameArguments(ARG0="x1", ARG1="x2", ARG2="x3", ARG3="x4")

# Step 2: Create the fitness function and individual class
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize fitness
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Step 3: Define fitness evaluation
def eval_func(individual):
    func = toolbox.compile(expr=individual)
    inputs = [[random.uniform(0, 1) for _ in range(4)] for _ in range(100)]
    outputs = []
    for inp in inputs:
        try:
            result = func(*inp)
            outputs.append(min(1.0, max(-1.0, result)))  # Clamp to [-1, 1]
        except (OverflowError, ZeroDivisionError):
            outputs.append(1.0)  # Penalize invalid outputs
    return sum(abs(o) for o in outputs),  # A simple fitness: minimize output magnitude

toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", eval_func)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
toolbox.register("select", tools.selTournament, tournsize=3)

# Step 4: Run the genetic programming algorithm
def main():
    random.seed(42)
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", min)
    stats.register("max", max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, stats=stats, halloffame=hof, verbose=True)

    print("Best individual:", hof[0])
    print("Fitness of the best individual:", hof[0].fitness.values)
    return pop, log, hof

if __name__ == "__main__":
    main()
