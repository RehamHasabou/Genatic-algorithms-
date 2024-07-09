import math
import random
import numpy as np
import matplotlib.pyplot as plt

cities=[(0,1),(0.009174316,0.995412849),(0.064220188,0.438073402),(0.105504591,0.594036699),
(0.105504591,0.706422024),(0.105504591,0.917431193),(0.185779816,0.454128438),(0.240825687,0.614678901),
(0.254587155,0.396788998),(0.38302753,0.830275235),(0.394495416,0.839449537),(0.419724769,0.646788988),
(0.458715603,0.470183489),(0.593922025,0.348509173),(0.729357795,0.642201837),(0.731651377,0.098623857),
(0.749999997,0.403669732),(0.770642198,0.495412842),(0.786697249,0.552752296),(0.811926602,0.254587155),
(0.852217125,0.442928131),(0.861850152,0.493004585),(0.869762996,0.463761466),(0.871559638,0),(0.880733941,0.08486239),
(0.880733941,0.268348623),(0.885321106,0.075688073),(0.908256877,0.353211012),(0.912270643,0.43470948)]

alpha = 1.0
beta = 2.0
evaporation_rate = 0.5
num_nodes=30
#pheromone = np.ones((num_nodes, num_nodes))/num_nodes

def distance(x1,y1,x2,y2):
    return math.sqrt((x2-x1)**2+(y2-y1)**2)

def calculate_distance():
    n=len(cities)
    dist = []
    for i in range(n):
       row = []
       for j in range(n):
           row.append(0)
       dist.append(row)  

    for i in range(n):
        for j in range(n):
            x1, y1 = cities[i]
            x2, y2 = cities[j]
            dist[i][j] = distance(x1, y1, x2, y2)
    return dist     

dis=calculate_distance()

def calculate_eta():
    n=len(dis)
    eta=[]
    for i in range(n):
       row = []
       for j in range(n):
           row.append(0)
       eta.append(row)  
    for i in range(n):
        for j in range(n):
            if (i!=j):
                x1, y1 = cities[i]
                x2, y2 = cities[j]
                eta[i][j] = 1 /distance(x1, y1, x2, y2)
    return eta    
eta=calculate_eta()

def nearest_neighbor(distance,start_city):
    n = len(distance)
    tour_length = 0
    if start_city is None:
       start_city = random.randint(0, n - 1) 
       tour=[start_city]
       visited = [start_city]  
    else:
        tour = [start_city]
        visited = [start_city]  
       
    for _ in range(n - 1):  
        current_city= tour[-1]
        nearest_dist = float('inf') 
        nearest_city= None
        for Next_city in range(n):
            if Next_city not in visited:  
                if distance[current_city][Next_city] < nearest_dist:
                    nearest_dist = distance[current_city][Next_city]
                    nearest_city= Next_city 
        tour.append(nearest_city)
        visited.append(nearest_city)
        tour_length += nearest_dist
    tour_length += distance[tour[-1]][start_city]
    tour.append(start_city)  

    return tour_length ,tour

#tour, tour_length = nearest_neighbor(dis,None)
#print("Tour:", tour)
#print("Tour Length:", tour_length)
def tour_length(tour):
    length, _ = nearest_neighbor(dis, tour[0])
    return length

def calculate_tau(num_nodes):
    tour_len,_ = nearest_neighbor(dis, None)
    tau_0 = 1 / (num_nodes * tour_len)
    tau=[]
    for i in range(num_nodes):
       row = []
       for j in range(num_nodes):
           row.append(tau_0)
       tau.append(row)  
    return tau
num_nodes=len(cities)
tau=calculate_tau(num_nodes)

current_city=random.randint(1, 28)
def next_city(current_city,visited,tau,eta,alpha,beta):
    probabilities = []
    sum_prob=0.0
    for city in range(num_nodes):
        if city not in visited:
            pheromone=tau[current_city][city]
            attractivness=eta[current_city][city]
            prob=(pheromone**alpha)*(attractivness**beta)
            probabilities.append(prob)
            sum_prob+=prob
        else:
            probabilities.append(0)
    
    probabilities=[p/sum_prob for p in probabilities]
    next_node = np.random.choice(range(num_nodes), p=probabilities)
    return next_node
#Next=next_city(current_city,visited, tau, eta, alpha, beta)       
def generate_ants():
    m = random.randint(1, 15)
    ants = []
    for _ in range(m):
        start_city = random.randint(0, num_nodes - 1)
        current_city = start_city
        visited = [current_city]
        while len(visited)< num_nodes:
            Next= next_city(current_city, visited, tau, eta, alpha, beta)
            visited.append(Next)
            current_city = Next
        visited.append(start_city)
        ants.append(visited)
    return ants   

def remove_cycles(tours):
    acyclic_tours = []
    for tour in tours:
        acyclic= []
        visited = []
        for city in tour:
            if city not in visited:
                acyclic.append(city)
                visited.append(city)
        acyclic.append(acyclic[0])
        acyclic_tours.append(acyclic)
    return acyclic_tours    

def update_pheromones(tau, acyclic, evaporation_rate):
    for i in range(len(tau)):
        for j in range(len(tau[i])):
            tau[i][j] *= (1 - evaporation_rate)
    for tour in acyclic:
        tour_len = tour_length(tour)
        n=len(tour)
        for i in range(n - 1):
            city1 = tour[i]
            city2 = tour[i + 1]
            tau[city1][city2] += 1 / tour_len 
    return tau

for _ in range(20):
    ants = generate_ants()
    acyclic_tours = remove_cycles(ants)
    tau = update_pheromones(tau, acyclic_tours, evaporation_rate)
    
def plot(cities, tours):
    x = [city[0] for city in cities]
    y = [city[1] for city in cities]

    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, color='red', label='Cities')

    for tour in tours:
        tour_x = [cities[i][0] for i in tour]
        tour_y = [cities[i][1] for i in tour]
        tour_x.append(tour_x[0])
        tour_y.append(tour_y[0])
        plt.plot(tour_x, tour_y, color='blue', linewidth=2)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('TSP Tours')
    plt.legend()
    plt.show()

ants = generate_ants()
acyclic_tours = remove_cycles(ants)
plot(cities, acyclic_tours)
