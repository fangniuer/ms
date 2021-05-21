from deap import algorithms, base, cma, creator, tools

import numpy as np

"""
The following definitions are optimize algorithms used in biological models whose file name is 'BioModels.py'.
"""

def CMAES(objective_function, x_initial, x_min, x_max, maxiter, verbose=True, **kwargs):
    
    '''
    
    Parameters
    ----------
    objective_function : callable
        The objective function to be minimized. Must be in the form f(x, **kwargs), where x is the argument in the form of a 1-D sequence and kwargs is a dictionary or mapping of any additional fixed parameters needed to completely specify the function.
    x_initial : sequence
        The initial values of x in the form of a 1-D sequence.
    x_min : sequence
        The lower bounds of x in the form of a 1-D sequence.
    x_max : sequence
        The upper bounds of x in the form of a 1-D sequence.
    maxiter : int
        The maximum of iterations.
    verbose : bool, optional
        Whether or not to log the statistics. The default is True.
    **kwargs : dictionary or mapping, optional
        Any additional fixed parameters needed to completely specify the objective function.
    
    Returns
    -------
    hof : class
        A class:`~deap.tools.HallOfFame` object that will contain the best individuals.
    logbook : class
        A class:`~deap.tools.Logbook` with the statistics of the evolution.
    history : dictionary
        A dictionary containing the records of four items in all iterations, in the form of list. The four items are population, the fitness values of population, the best of individual and the best value of fitness.
    
    '''
    
    def checkBounds(x_min, x_max):
        def decorator(func):
            def wrapper(*args, **kargs):
                offspring = func(*args, **kargs)
                for child in offspring:
                    # Check lower bounds and upper bounds for parameters.
                    for i in range(len(child)):
                        child[i] = min(child[i], x_max[i])
                        child[i] = max(child[i], x_min[i])
                return offspring
            return wrapper
        return decorator
    
    if hasattr(creator, 'FitnessMin'):
        pass
    else:
        creator.create("FitnessMin", base.Fitness, weights = (-1.0,))

    if hasattr(creator, 'Individual'):
        pass
    else:
        creator.create("Individual", list, fitness = creator.FitnessMin)
    
    toolbox = base.Toolbox()
    toolbox.register("evaluate", objective_function, **kwargs)
    
    # The CMA-ES algorithm converge with good probability with those settings
    cma_es = cma.Strategy(centroid = x_initial, sigma = 2 * len(x_initial))
    
    toolbox.register("generate", cma_es.generate, creator.Individual)
    toolbox.register("update", cma_es.update)
    toolbox.decorate("generate", checkBounds(x_min, x_max))
    
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # pop, logbook = algorithms.eaGenerateUpdate(toolbox, ngen = maxiter, stats = stats,
    #                                             halloffame = hof, verbose = True)
    
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    
    history = {'populations': [],
               'fitnesses': [],
               'best_xs': [],
               'best_fs': []}
    
    for gen in range(maxiter):
        # Generate a new population
        pop = toolbox.generate()
        # Evaluate the individuals
        fitnesses = toolbox.map(toolbox.evaluate, pop)
        
        fits = []
        pops = []
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
            pops.append(ind)
            fits.append(fit[0])
    
        if hof is not None:
            hof.update(pop)
    
        # Update the strategy with the evaluated individuals
        toolbox.update(pop)
    
        record = stats.compile(pop) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(pop), **record)
        if verbose:
            print(logbook.stream)
        
        history['populations'].append(pops)
        history['fitnesses'].append(fits)
        history['best_xs'].append(hof[0])
        history['best_fs'].append(hof[0].fitness.values[0])
    
    return hof, logbook, history
