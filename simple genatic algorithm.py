import random
import matplotlib.pyplot as plt


N = 20
n = 5
population = []
fitness = []
probablity = []
cumulative=[]
result_cumulative=[]
new_population=[]

# genrate individual
def genrate_chromosomes(n):
    b1 = ""
    for i in range(n):
        binary = str(random.randint(0, 1))
        b1 += binary
    return (b1)

# Evaluate the fitness of each individua
def evaluate_fitness(population):
    for i in range(N):
        individual = genrate_chromosomes(n)
        population.append(individual)
        target = sum(int(index) for index in individual)
        fitness.append(target)
    return fitness
  
fitness = evaluate_fitness(population)  
print("")

# Print each individual
for i, ind in enumerate(population):
    print(f"Individual {i+1}: {ind}")

# Print the fitness of each individual
print("")
for i, fit in enumerate(fitness):
    print(f"The fitness of individual {i+1}: {fit}")

# generates probabilitie for each individual
def calculate_probability(fitness):
    for i in range(N):
        pi = fitness[i]/sum(fitness)
        probablity.append(pi)
    return probablity  
 
#print probabilities for each individual
prob=calculate_probability(fitness)
print("")
for i, p in enumerate(prob):
    print(f"The probability of individual {i+1}: {p}")
    
#cumulative probabilities for each individua
cum=0
def calculate_cumulative(probability):
    cum=0
    for i in range(N):
        cum=probablity[i]+cum
        cumulative.append(cum)  
    return cumulative    
result=calculate_cumulative(probablity)
print("")
#print cumulative probabilities for each individua
for i, cum in enumerate(result):
    print(f"The cumulative probability of  {i+1}: {cum}")

#selection of the best individuals
selected_indvidual=[]
N=len(population)
for i in range(N):
     random_num=random.random()
     for j in range(N):
         if random_num <= cumulative[j]:
             selected_indvidual.append(population[j])
             break
#print selection of the best individuals
for i, select in enumerate(selected_indvidual):
    print(f"The selected individual  {i+1}: {select}")

#function for cross over
def onepoint_crossover(pair):
    
    split_point = random.randint(1, len(str(pair[0])) - 1)
    offspring1 = int(str(pair[0])[:split_point] + str(pair[1])[split_point:])
    offspring2 = int(str(pair[1])[:split_point] + str(pair[0])[split_point:])
    return offspring1, offspring2

pcross=0.6
def crossover(population, pcross):
    pairs = []
    for i in range(0, len(population), 2):
        pair = population[i], population[i+1]
        pairs.append(pair)
    
    offspring = []
    for pair in pairs:
        num = random.random()
        if num < pcross:
            offspring1, offspring2 = onepoint_crossover(pair)
            offspring.extend([offspring1, offspring2])
        else:
            offspring.extend(pair)
            
    return offspring

#print cross over
offspring = crossover(selected_indvidual,pcross)
print(" ")
print ("Cross over result :")
for i in range(len(offspring)):
   print(f"I{i+1}: {offspring[i]}")

#fnction for mutation
pmut = 0.05
def Bit_flip_mutation(population,pmut):
  
    for individual in population :
       mutation= ""
       individual = str(individual)
       for index in individual :
             num = random.random()
             #print(num)
             if num < pmut:
                 if index == "0":
                    mutation += "1"
                 else:
                     mutation += "0"
             else:
                 mutation += index
       new_population.append(int(mutation))
    return new_population

print(" ")
print ("The result of mutation :")
mutat=Bit_flip_mutation(population, pmut)
for i, mute in enumerate(mutat):
    print(f"the mutation result {i+1}: {mute}")
 
def elitism(population, fitness):
    the_best = [(population[i], fitness[i]) for i in range(N)]
    the_best.sort(key=lambda x: x[1], reverse=True)
    return the_best[:2]

number_generation=100
def generation(N,number_generation,n,pcross,pmut):
    prob = []
    cum = []
    new_population = []
    fitness= []
    population=[]
    best_hist = []
    avg_fitness_hist = []
    select=[]
    elite=[]
    for i in range(number_generation):
        fitness=evaluate_fitness(population)
        best_fitness = max(fitness)
        average_fitness = sum(fitness) / N
        best_hist.append(best_fitness)
        avg_fitness_hist.append(average_fitness)
        elite = elitism(population , fitness)
        prob = calculate_probability(fitness)
        cum = calculate_cumulative(prob)
        offspring = crossover(selected_indvidual , pcross)
        new_population = Bit_flip_mutation(offspring, pmut)
        population = new_population
        population.extend(elite) 
        population.extend(new_population[:N-2]) 
         
    print("Final Population:")
    for i, ind in enumerate(population[:20]):
        print(f"Individual {i+1}: {ind}")
    return best_hist, avg_fitness_hist
print(" ")     
print (generation(N, number_generation, n, pcross, pmut))
best_hist, avg_fitness_hist= generation(N, number_generation, n, pcross, pmut)

num_runs = 10
results = []

for i in range(num_runs):
    best_hist, avg_fitness_hist = generation(N, number_generation, n, pcross, pmut)
    results.append((best_hist, avg_fitness_hist))

# Plot the results of each run
plt.figure(figsize=(12, 6))
for i in range(num_runs):
    plt.plot(results[i][0], label=f'Run {i+1}')

# Plot of fitnes
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

fitness_list, avg_fitness_list = generation(N, number_generation, n, pcross, pmut)

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

