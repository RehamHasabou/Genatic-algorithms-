import numpy as np
import random
import matplotlib.pyplot as plt

n=2
N=20
R_min=-2 
R_max=2

def max_fun(x1,x2):
     return 8 - (x1 + 0.0317) ** 2 + x2 ** 2
 
def initialize_population(N, n, R_min, R_max):
    population = np.zeros((N, n))
    for i in range(N):
        for j in range(n):
            population[i, j] = np.random.uniform(R_min, R_max)
    return population

population= initialize_population(N, n, R_min, R_max)
print("Population :",population)
alpha = 0.6

def arithmetic_crossover(alpha):
    parent1, parent2 = np.split(population, 2, axis=1)
    parent1.flatten()
    parent2.flatten()
    child1 = (alpha * parent1) + ((1 - alpha) * parent2) 
    child2 = (alpha * parent2) + ((1 - alpha) * parent1)
    cross = np.concatenate((child1, child2), axis=1)
    return cross
cross=arithmetic_crossover(alpha)
print("-----------------------")
print("The crossover :",cross)

pmut=0.05
sigma=0.5

def gaussian_mutation(cross,sigma,pmut):
    crossover= np.copy(cross)
    for i in range(N):
       for j in range(n):
          num=np.random.rand()
          if(num<pmut):
           crossover[i,j]+=np.random.normal(loc=0, scale=sigma)
    return crossover       

mutated_individual = gaussian_mutation(cross,sigma,pmut)
print("-------------------------")
print("Mutated individual:",mutated_individual)
def evaluate_fitness(pop):
    fitness=[]
    for individual in pop:
        x1, x2 = individual
        result = max_fun(x1, x2)
        fitness.append(result)
    return fitness 
fitness =evaluate_fitness(population)

tournment_size=4
def tournment_selection(pop, tournment_size):
    selected_parents = []
    for i in range(len(pop)):
       subset=random.sample(list(pop), tournment_size)
       selected=max(evaluate_fitness(subset))
       selected_parents.append(selected)
    return selected_parents

print("-------------------------")
selected_individuals = tournment_selection(mutated_individual, tournment_size)
print("The selection of indvidual :",selected_individuals)

def elitism(pop, fitness):
    elite = [(pop[i], fitness[i]) for i in range(len(pop))]
    elite.sort(key=lambda x: x[1], reverse=True)
    return elite[:2]

number_generation=100
def generation(N, n, alpha, pmut):
    best_hist = []
    avg_fitness_hist = []
    elitsm=[]
    for i in range(number_generation):
        population = initialize_population(N, n, R_min, R_max)
        fitness = evaluate_fitness(population)
        elite=elitism(population, fitness)
        elitsm.append(elite)
        best_fitness = max(fitness)
        average_best_fitness = sum(fitness) / N
        best_hist.append(best_fitness)
        avg_fitness_hist.append(average_best_fitness)
        crossover = arithmetic_crossover(alpha)
        mutation = gaussian_mutation(crossover, sigma, pmut)
        new_population = tournment_selection(mutation, tournment_size)
        population = new_population.copy()
        population.extend(new_population[:N - 2])
    return best_hist, avg_fitness_hist     
                                  
best_hist, avg_fitness_hist = generation(N, n, alpha, pmut)
print("--------------------")
print("The best fitness:", best_hist)
print("--------------------")
print("The average of beat fitness:", avg_fitness_hist)


num_runs = 10
results = []

for i in range(num_runs):
    best_hist, avg_fitness_hist = generation(N, n, alpha, pmut)
    results.append((best_hist, avg_fitness_hist))

# Plot the results of each run
plt.figure(figsize=(12, 6))
for i in range(num_runs):
    plt.plot(results[i][0], label=f'Run {i+1}')

# Plot of fitness
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Genetic Algorithm Performance')
plt.legend()
plt.grid(True)
plt.show()

# Plot of average fitness
plt.figure(figsize=(12, 6))

for i in range(num_runs):
    plt.plot(results[i][1], label=f'Run {i+1} (Average Fitness)')

plt.xlabel('Generation')
plt.ylabel('Average Fitness')
plt.title('Genetic Algorithm Performance (Average Fitness)')
plt.legend()
plt.grid(True)
plt.show()

fitness_list, avg_fitness_list = generation(N, n, alpha, pmut)

# Plot the histograms
plt.figure(figsize=(12, 6))

# Plot histogram of fitness
plt.subplot(1, 2, 1)
plt.hist(fitness_list, bins=10, edgecolor='black')
plt.xlabel('Fitness')
plt.ylabel('Frequency')
plt.title('Distribution of Fitness')

# Plot histogram of average fitness
plt.subplot(1, 2, 2)
plt.hist(avg_fitness_list, bins=10, edgecolor='black')
plt.xlabel('Average Fitness')
plt.ylabel('Frequency')
plt.title('Distribution of Average Fitness')

plt.tight_layout()
plt.show()




    
    
    
    
    