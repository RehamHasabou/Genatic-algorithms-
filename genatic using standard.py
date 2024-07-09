import random 
import math
import numpy as np
import matplotlib.pyplot as plt

pop=[]
n=6
N=100
def genrate_chromosomes(n):
    pop=[[random.randint(0, 1) for i in range(n)] for i in range(N)]
    return pop
def max_fun(x1,x2):
     return 8 - (x1 + 0.0317) ** 2 + x2 ** 2
def constrain(x1,x2):
    return x1+x2-1

def standard_decoder(pop, variable_min, variable_max):
    decoded_values = []
    for indv in range(len(pop)):
        length =len(pop[indv])
        first_variable = pop[indv][:length //2]  
        second_variable = pop[indv][length //2:]  

        solutions = 2 ** (length // 2)
        sum = 0
        for i in range(length  // 2):
            sum += first_variable[i] * (2 ** ((length  // 2) - i - 1))

        real_value = variable_min + (sum /solutions) * (variable_max - variable_min)
        decoded_values.append(math.ceil(real_value))

        solutions = 2 ** (length // 2)
        sum = 0
        for i in range(length // 2):
            sum += second_variable[i] * (2 ** ((length // 2) - i - 1))

        real_value = variable_min + (sum / solutions) * (variable_max - variable_min)
        decoded_values.append(math.ceil(real_value))
    
    return decoded_values

def evaluate_standerd_fitness(variable_min, variable_max):
    decoded_values = standard_decoder(pop, variable_min, variable_max)
    standerd_fitness = []
    for i in range(0, len(decoded_values), 2):
        first_variable = decoded_values[i]
        second_variable = decoded_values[i + 1]
        result= max_fun(first_variable, second_variable)
        #result= max_fun(first_variable, second_variable)-constrain(first_variable,second_variable)
        standerd_fitness.append(result)
        #print("Individual real value and its fitness:")
        #print(first_variable,second_variable, result)
    return standerd_fitness
'''
pop = genrate_chromosomes(n)

standerd_fitness = evaluate_standerd_fitness(-2,2)  
print("")

# Print each individual
for i, ind in enumerate(pop):
    print(f"Individual {i+1}: {ind}")

# Print the fitness of each individual
print("----------")
for i, fit in enumerate(standerd_fitness):
    print(f"The fitness of individual {i+1}: {fit}")
print("----------")  
'''
def gray_decoding(pop, variable_min, variable_max):
    gray_values = []
    for indv in range(len(pop)):
        length =len(pop[indv])
        first_variable = pop[indv][:length //2]  
        second_variable = pop[indv][length //2:]  

        solutions = 2 ** (length // 2)
        sum = 0
        for i in range(length  // 2):
            sum += (first_variable[i]%2 )* (2 ** ((length  // 2) - i - 1))

        real_value = variable_min + (sum /solutions) * (variable_max - variable_min)
        gray_values.append(math.ceil(real_value))

        solutions = 2 ** (length // 2)
        sum = 0
        for i in range(length // 2):
            sum += (second_variable[i]%2)* (2 ** ((length // 2) - i - 1))

        real_value = variable_min + (sum / solutions) * (variable_max - variable_min)
        gray_values.append(math.ceil(real_value))
    return gray_values

#print(gray_decoding(pop, -2, 2))

def evaluate_gray_fitness(variable_min, variable_max):
    gray_values = gray_decoding(pop, variable_min, variable_max)
    gray_fitness = []
    for i in range(0, len(gray_values), 2):
        first_variable = gray_values[i]
        second_variable = gray_values[i + 1]
        #result= max_fun(first_variable, second_variable)
        result= max_fun(first_variable, second_variable)-constrain(first_variable,second_variable)
        gray_fitness.append(result)
        #print("Individual real value and its gray_fitness:")
        #print(first_variable,second_variable, result)
    return gray_fitness

pop = genrate_chromosomes(n)
gray_fitness = evaluate_gray_fitness(-2,2)  
print("")

# Print each individual
for i, ind in enumerate(pop):
    print(f"Individual {i+1}: {ind}")

# Print the fitness of each individual
print("----------")
for i, fit in enumerate(gray_fitness):
    print(f"The fitness using gray decoding {i+1}: {fit}")
    #print(f"The fitness using standerd decoding {i+1}: {fit}")
print("----------")    

sp=1.5
def linear_rank(pop,sp):
  pop_size = len(pop)
  #ranks = np.array(standerd_fitness).argsort().argsort() + 1
  ranks = np.array(gray_fitness).argsort().argsort() + 1
  #print("Ranks:",ranks)
  #print("----------")
  rank_fitness = [(2-sp) + 2 * (sp - 1) * (rank-1)/(pop_size-1) for rank in ranks]
  return rank_fitness

linear=linear_rank(pop, sp)
for i, lin in enumerate(linear):
    print(f"Ranke fitness {i+1}: {lin}")

print("----------") 
# generates probabilitie for each individual
def calculate_probability(rank_fitness):
    probability=[]
    for i in range(len(pop)):
        pi = rank_fitness[i]/sum(rank_fitness)
        probability.append(pi)
    return probability  

prob= calculate_probability(linear_rank(pop, sp))
def calculate_cumulative(probability):
    cumulative=[]
    cum=0
    for i in range(len(probability)):
        cum+=probability[i]
        cumulative.append(cum)  
    return cumulative 

cumulative = calculate_cumulative(prob)
#print cumulative probabilities for each individual
for i, cum in enumerate(cumulative):
    print(f"The cumulative probability of individual  {i+1}: {cum}")   

def selection(pop, cumulative):
    selected_individual = []
    N = len(pop)
    for i in range(N):
        random_num = random.random()
        for j in range(N):
            if random_num <= cumulative[j]:
                selected_individual.append(pop[j])
                break
    return selected_individual

def onepoint_crossover(pair):
    split_point = random.randint(1, len(pair[0]) - 1)
    offspring1 = (pair[0])[:split_point] + (pair[1])[split_point:]
    offspring2 = (pair[1])[:split_point] +pair[0][split_point:]
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
offspring = crossover(selection(pop,cumulative),pcross)
print("----------") 
print ("Cross over result :")
for i in range(len(offspring)):
   print(f"I{i+1}: {offspring[i]}")

#fnction for mutation
pmut = 0.05
def Bit_flip_mutation(pop,pmut):
    new_population=[]
    for individual in pop:
       mutation=[]
       for index in individual :
             num = random.random()
             #print(num)
             if num < pmut:
                if index == 0:
                    mutation.append(1)
                else:
                    mutation.append(0)
             else:
                mutation.append(index)
       new_population.append(mutation)
    return new_population

print("----------") 
print ("The result of mutation :")
mutat=Bit_flip_mutation(pop, pmut)
for i, mute in enumerate(mutat):
    print(f"the mutation result {i+1}: {mute}")
    
def elitism(pop, fitness):
    elite = [(pop[i], fitness[i]) for i in range(len(pop))]
    elite.sort(key=lambda x: x[1], reverse=True)
    return elite[:2]

number_generation=100
def generation(N, n, pcross, pmut):
    best_hist = []
    avg_fitness_hist = []
    for i in range(number_generation):
        #fitness = evaluate_standerd_fitness(-2, 2)
        fitness = evaluate_gray_fitness(-2, 2)
        best_fitness = max(fitness)
        average_fitness = sum(fitness) / N
        best_hist.append(best_fitness)
        avg_fitness_hist.append(average_fitness)
        elite=elitism(pop, fitness)
        prob = calculate_probability(fitness)
        cum = calculate_cumulative(prob)
        select = selection(pop, cum)
        offspring = crossover(select, pcross)
        new_population = Bit_flip_mutation(offspring, pmut)    
        population = new_population
        population.extend(elite) 
        population.extend(new_population[:N - 2])
        '''
    print("Final Population:")
    for i, ind in enumerate(population[:N]):
        print(f"Individual {i+1}: {ind}")
    '''    
    return best_hist, avg_fitness_hist

print("----------") 
print(generation(N, n, pcross, pmut))
best_hist_results, avg_fitness_hist_results = generation(N, n, pcross, pmut)
#print("elitism",elite)
num_runs = 10
results = []

for i in range(num_runs):
    best_hist, avg_fitness_hist = generation(N, n, pcross, pmut)
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

fitness_list, avg_fitness_list = generation(N, n, pcross, pmut)

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