import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from math import *
from mesa import Model, Agent, DataCollector
from mesa.time import SimultaneousActivation
from tqdm.notebook import trange
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.img_tiles import OSM

class Lieu():
    R_TERRE = 6373.0
    def __init__(self , id : int, longitude : float, latitude : float):
        self.id = id
        self.longitude = longitude
        self.latitude = latitude

    def distance(self, other ) -> float:
        
        dlat = radians(other.latitude - self.latitude)
        dlon = radians(other.longitude - self.longitude)
        a = sin(dlat / 2)**2 + cos(self.latitude) * cos(other.latitude) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return self.R_TERRE * c

    def __str__(self):
        return f'Lieu {self.id}'
    
class Client(Lieu):
    def __init__(self, id : int, longitude : float, latitude : float,  demande : float):
        super().__init__(id, longitude, latitude)
        self.demande = demande
    
    def __str__(self):
        return f'{self.id}'
    
    def __repr__(self):
        return f'{self.id}'
    
class Depot(Lieu):
    def __init__(self, id : int, longitude : float, latitude : float):
        super().__init__(id, longitude, latitude)

    def __str__(self):
        return f'Depot {self.id}'

class Vehicule() :
    def __init__(self, id : int, capacite : float):
        self.id = id
        self.capacite = capacite

    def __str__(self):
        return str(self.id)

    def __repr__(self):
        return str(self.id)
    
    def mesure_load(self, actions : list[Lieu]) -> float  :
        load =self.capacite
        for action in actions :
            if isinstance(action, Client) :
                load -= action.demande
            elif isinstance(action, Depot) :
                load = self.capacite
        return load

    def mesure_distance(self, actions : list[Lieu]) -> float :
        if len(actions) <=1 : return 0
        d = 0
        for i in range(len(actions)-1) :
            d += actions[i].distance(actions[i+1])
        return d





# SOLUTIONS


class SolutionFinale():
    def __init__(self, schedule : dict[Vehicule, list[Lieu]], probleme): 
        self.schedule = schedule
        self.probleme = probleme

    @property   
    def nb_vehicles(self) -> int :
        return len(self.schedule)

    def impose_capacity_constraint(self):
        for vehicule in self.schedule :
            N= len(self.schedule[vehicule])
            if N == 0 : pass
            done = []
            to_do = self.schedule[vehicule]
            while to_do :
                next = to_do.pop(0)
                if isinstance(next, Client) and next.demande > vehicule.mesure_load(done) :
                    done.append(self.probleme.depot)
                done.append(next)
            self.schedule[vehicule] = done


    def score (self) -> float :
        self.impose_capacity_constraint()
        for vehicule in self.schedule :
            if len(self.schedule[vehicule]) >0 and not isinstance(self.schedule[vehicule][-1], Depot) : 
                self.schedule[vehicule].append(self.probleme.depot)
            if len(self.schedule[vehicule]) >0 and not isinstance(self.schedule[vehicule][0], Depot) :
                self.schedule[vehicule].insert(0, self.probleme.depot)
        c = self.nb_vehicles * self.probleme.cout_vehicule
        distances = [vehicule.mesure_distance(self.schedule[vehicule]) for vehicule in self.schedule]
        c += sum(distances) +max(distances)*self.nb_vehicles
        return c

    def score_2(self) -> float:
        self.impose_capacity_constraint()
        if self._score is not None:
            return self._score  # Retourner le score précédemment calculé si disponible

        total_cost = self.nb_vehicles * self.probleme.cout_vehicule

        for vehicle, path in self.schedule.items():
            # Créer une copie temporaire de l'itinéraire pour le véhicule
            temp_path = [self.probleme.depot]+path[:]  if not isinstance(path[0], Depot) else path[:]
            if len(path) > 0 and not isinstance(path[-1], Depot):
                temp_path.append(self.probleme.depot)

            # Calculer la distance pour le chemin temporaire
            distance = vehicle.mesure_distance(temp_path)
            total_cost += distance

        self._score = total_cost  # Mémoriser le score calculé
        return self._score

    def __str__(self):
        return str(self.schedule)
    
    def plot(self):
        self.impose_capacity_constraint()
        print(self.nb_vehicles)
        print()
        for vehicule in self.schedule :
            if len(self.schedule[vehicule]) > 1 and not isinstance(self.schedule[vehicule][-1], Depot) : 
                self.schedule[vehicule].append(self.probleme.depot)
            if len(self.schedule[vehicule]) > 1 and not isinstance(self.schedule[vehicule][0], Depot) :
                self.schedule[vehicule].insert(0, self.probleme.depot)
        fig = plt.figure(figsize = (7,7))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        imagery = OSM()
        ax.add_image(imagery, 9)
        for vehicule in self.schedule :
            if len(self.schedule[vehicule]) > 1 :
                ax.plot([l.longitude for l in self.schedule[vehicule]], [l.latitude for l in self.schedule[vehicule]])
                #for l in self.schedule[vehicule] :
                    #plt.text(l.longitude , l.latitude, str(l.id), horizontalalignment='center',verticalalignment='center')
        ax.plot(self.probleme.depot.longitude, self.probleme.depot.latitude, 'ro')
        plt.figure()
        i = 0
        for vehicule in self.schedule :
            if len(self.schedule[vehicule])>1 :
                origin = self.schedule[vehicule][0]
                time = 0
                #plt.text(0,len(self.schedule)-i, self.schedule[vehicule][0], horizontalalignment='center',verticalalignment='center')
                for j in range(1,len(self.schedule[vehicule])) :
                    plt.plot([time, time + origin.distance(self.schedule[vehicule][j])], [len(self.schedule)-i-j/len(self.schedule[vehicule]),len(self.schedule)-i-j/len(self.schedule[vehicule])])
                    #plt.text(time + origin.distance(self.schedule[vehicule][j]), len(self.schedule)-i-(j+0.5)/len(self.schedule[vehicule]), self.schedule[vehicule][j], horizontalalignment='center',verticalalignment='center')
                    time += origin.distance(self.schedule[vehicule][j])
                    origin = self.schedule[vehicule][j]
                i+=1
        plt.figure()


class Solution():
    def __init__(self, chromosome : list[Client], N_vehicules : int, probleme):
        self.chromosome = chromosome
        self.N_vehicules = N_vehicules
        self.probleme = probleme
    
    def __str__(self):
        return f"{[str(l) for l in self.chromosome]} by {self.N_vehicules} vehicles"

    def generate_neighbors(self, num_neighbors: int = 10):
        
        '''
        Function that generates direct neighbors from the solution
        
        Parameters:
            num_neighbors (int): Number of neighbor to generate
            
        Returns:
            neighbors (list[Client]): List of random neighbors
        '''
        
        neighbors: list[Solution] = []
        for _ in range(num_neighbors):
            new_chromosome: list[Client] = []
            i1: int = rd.randint(0, len(self.chromosome) - 1)
            i2: int = rd.randint(0, len(self.chromosome) - 1)
            for i in range(len(self.chromosome)):
                if i == i1:
                    new_chromosome.append(self.chromosome[i2])
                elif i == i2:
                    new_chromosome.append(self.chromosome[i1])
                else:
                    new_chromosome.append(self.chromosome[i])
                neighbors.append(Solution(new_chromosome, self.N_vehicules, self.probleme))
        return neighbors

    def mutate(self, p):
        if rd.random() > p : 
            i1 = rd.randint(0,len(self.chromosome)-1)
            i2 = rd.randint(0,len(self.chromosome)-1)
            self.chromosome[i1], self.chromosome[i2] = self.chromosome[i2], self.chromosome[i1]
        if rd.random()> p :
            if rd.random() > 0.5 : 
                self.N_vehicules += 1
            else : 
                if self.N_vehicules>1:
                    self.N_vehicules -= 1
        return self
        
    def cross(self, other , p):
        if self.probleme != other.probleme :
            print(f'p1 : {self.probleme}, {other.probleme}')
            raise ValueError("The two solutions must be from the same problem")
        if rd.random() > p :
            a = rd.randint(0,len(self.chromosome))
            fils1 = [None]*len(self.chromosome)
            fils2 = [None]*len(self.chromosome)
            fils3 = [None]*len(self.chromosome)
            fils4 = [None]*len(self.chromosome)
            fils1[:a]=self.chromosome[:a]
            fils2[:a]=other.chromosome[:a]
            fils3[a:]=self.chromosome[a:]
            fils4[a:]=other.chromosome[a:]
            i=0
            for t in other.chromosome :
                if t not in fils1 :
                    fils1[a+i]=t
                    i+=1
            i=0
            for t in self.chromosome :
                if t not in fils2 :
                    fils2[a+i]=t
                    i+=1
            i=0
            for t in other.chromosome :
                if t not in fils3 :
                    fils3[i]=t
                    i+=1
            i=0
            for t in self.chromosome :
                if t not in fils4 :
                    fils4[i]=t
                    i+=1

            return [Solution(fils1, self.N_vehicules, self.probleme), Solution(fils2, other.N_vehicules, self.probleme), Solution(fils3, self.N_vehicules, self.probleme), Solution(fils4, other.N_vehicules, self.probleme)]
        else : 
            return [self, other]
    @property
    def final(self):
        vehicules = [Vehicule(i, self.probleme.capacite_vehicule) for i in range(self.N_vehicules)]
        res = {v : [] for v in vehicules}
        for i, c in enumerate(self.chromosome) :
            res[vehicules[i%self.N_vehicules]].append(c)
        res = {key : res[key] for key in res if len(res[key])>0}
        return SolutionFinale(res, self.probleme)


    def score(self) -> float :        
        return self.final.score()
    
    
    
    
# MODELE

class SMA_VRP(Model):
    def __init__(self, clients : list[Client], depot : Depot, cout_vehicule : float, capacite_vehicule : float):
        super().__init__()
        self.clients = clients
        self.cout_vehicule = cout_vehicule
        self.depot = depot
        self.capacite_vehicule = capacite_vehicule
        self.good_solution_pool : list[Solution] = []
        self.schedule = SimultaneousActivation(self)
        self.datacollector = DataCollector(
            model_reporters={"best_solution": lambda m :m.good_solution_pool[0].score() if len(m.good_solution_pool)>0 else None, 'pool_size': lambda m : len(m.good_solution_pool)}, agent_reporters={"Score": lambda a : a.best_score}
        )

    def gen_random_solution(self, n_transports: int):
        """
        Generate a random solution for the given number of transports.

        Parameters:
            n_transports (int): The number of transports to consider.

        Returns:
            Solution: A randomly generated solution using the clients and number of transports.
        """
        rd.shuffle(self.clients)
        return Solution(self.clients, n_transports, self)
    
    def add_agent(self, agent):
        self.schedule.add(agent)

    def insert_solution_in_pool(self, solution : Solution):
        """
        A function that inserts a solution into the good_solution_pool list in sorted order based on the solution score.

        Parameters:
            solution (Solution): The solution object to be inserted into the pool.

        Returns:
            None
        """
        s = solution.score()
        self.good_solution_pool.append(solution)
        self.good_solution_pool.sort(key=lambda sol: sol.score())
        print('list sorted, best score = ', self.good_solution_pool[0].score())
        return

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        
        
# AGENTS

class myAgent(Agent):
    def __init__(self, unique_id: int, model: SMA_VRP, pop_size: int):
        super().__init__(unique_id, model)
        self.population : list[Solution] = sorted([self.model.gen_random_solution(rd.randint(2, 10)) for _ in range(pop_size)], key = lambda x : x.score())
        self.best_score = min(self.population, key = lambda x : x.score()).score()
        
    def fetch_better_solutions_from_pool(self):
        """
        Fetches better solutions from the solution pool and updates the population accordingly.
        """
        to_get = [sol for sol in self.model.good_solution_pool if sol.score() < self.best_score]
        N = len(self.population)
        self.population = sorted(to_get + self.population, key = lambda x : x.score())
        self.population = self.population[:N]

    def push_solution_in_pool(self, sol : Solution):
        """
        A function that pushes a solution into the pool.

        Parameters:
            sol (Solution): The solution to be inserted into the pool.
        
        Returns:
            None
        """
        self.model.insert_solution_in_pool(sol)
        
    def step(self):
        pass
    
    
class GeneticAgent(myAgent):
    def __init__(self, unique_id: int, model: SMA_VRP, pop_size : int, Pcross : float, Pmut : float):
        """
        Initialize the VRP agent with a unique ID, model, population size, crossover probability, and mutation probability.
        """
        super().__init__(unique_id, model, pop_size)
        self.Pmut = Pmut
        self.Pcross = Pcross

    def step(self):
        """
        Generate the next generation of solutions based on the current population. 
        This function sorts the population, selects the top individuals, performs crossover and mutation, 
        updates the best score, and collects data for the model. 
        """
        self.fetch_better_solutions_from_pool()
        S = sorted(self.population, key = lambda x : x.score())
        S = S[:len(self.population)]
        nex_gen : list[Solution]= []
        for p1, p2 in zip(S[::2], S[1::2]):
            nex_gen.extend(p1.cross(p2, self.Pcross))
        self.population : list [Solution] = sorted([x.mutate(self.Pmut) for x in nex_gen], key = lambda x : x.score())[:len(self.population)]
        self.best_score = self.population[0].score()
        self.push_solution_in_pool(self.population[0])

        
class RSAgent(myAgent):
    
    def __init__(self, unique_id: int, model: SMA_VRP, pop_size: int, t0: float, cooling: float, nb_iter: int):
        
        '''
        Create an instance of RSAgent
        
        Parameters:
            t0 (float): initial temperature
            cooling (float): cooling rate
            nb_iter (int): maximal number of iterations of one step
        '''
        
        super().__init__(unique_id, model, pop_size)
        self.t = t0
        self.cooling = cooling
        self.max_iter = nb_iter
        
        
    def step(self):
        
        '''
        Makes the agent interact with the pool
        '''
        
        self.fetch_better_solutions_from_pool() #Update the population
        self.population = sorted(self.population, key = lambda x : x.score())
        
        #Parameters for the algorithm
        s1: Solution = self.population[0]
        s2: Solution = s1
        new_cycle: bool = True
        t: float = self.t
        nb_iter: int = 0
        
        while new_cycle and nb_iter < self.max_iter:
                
            nb_iter += 1
            new_cycle = False
            
            #Calculate the difference
            s2 = s2.generate_neighbors(1)[0]
            f1: float = s1.score()
            f2: float = s2.score()
            df: float = f1 - f2
            
            if df < 0:
                s1 = Solution(s2.chromosome, s2.N_vehicules, s1.probleme)
                new_cycle = True
            else: 
                #Probability 
                prob: float = exp(-df / t)
                q: float = rd.random()
                if q < prob:
                    s1 = Solution(s2.chromosome, s2.N_vehicules, s1.probleme)
                    new_cycle = True
                else: 
                    s2 = Solution(s1.chromosome, s1.N_vehicules, s1.probleme)
                    
            #Reduce the temperature
            t *= (1 - self.cooling)
            
        self.t = t
            
        #Update the attributes and the pool
        self.population.insert(0, Solution(s1.chromosome, s1.N_vehicules, s1.probleme))
        self.population.pop()
        self.best_score = self.population[0].score()
        self.push_solution_in_pool(self.population[0])
        
        
class TabouAgent(myAgent):
    
    def __init__(self, unique_id: int, model: SMA_VRP, pop_size: int, tabu_size: int = 10, neighbors: int = 10):
        super().__init__(unique_id, model, pop_size)
        self.nb_neighbors = neighbors
        self.visited = []
        self.size = tabu_size
        
    def step(self):
        
        self.fetch_better_solutions_from_pool() #Update the population
        self.population = sorted(self.population, key = lambda x : x.score())
        
        s1: Solution = self.population[0]
        s2: Solution = Solution(s1.chromosome, s1.N_vehicules, s1.probleme)
        
        neighbors = s2.generate_neighbors(self.nb_neighbors)
        neighbors = sorted(neighbors, key = lambda x: x.score())
        
        found = False
        for neighbor in neighbors:
            if neighbor not in self.visited and (s1 is None or neighbor.score() < s1.score()):
                s2 = neighbor
                if s2.score() < s1.score():
                    s1 = Solution(s2.chromosome, s2.N_vehicules, s2.probleme)
                found = True
                
        if found:
            self.visited.append(Solution(s2.chromosome, s2.N_vehicules, s2.probleme))
            if len(self.visited) > self.size:
                self.visited.pop(0)
                
        self.population.insert(0, Solution(s1.chromosome, s1.N_vehicules, s1.probleme))
        self.population.pop()
        self.best_score = self.population[0].score()
        self.push_solution_in_pool(self.population[0])
