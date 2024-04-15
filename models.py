import matplotlib.pyplot as plt
import random as rd
from math import *
from mesa import Model, Agent, DataCollector
from mesa.time import SimultaneousActivation
import cartopy.crs as ccrs
from cartopy.io.img_tiles import OSM


class Lieu:
    R_TERRE = 6373.0

    def __init__(self, location_id: int, longitude: float, latitude: float):
        """
        Initializes the location with the given ID, longitude, and latitude.

        Parameters:
            location_id (int): The ID of the location.
            longitude (float): The longitude coordinate of the location.
            latitude (float): The latitude coordinate of the location.
        """
        self.id = location_id
        self.longitude = longitude
        self.latitude = latitude

    def distance(self, other : 'Lieu') -> float:
        """
        Calculate the great-circle distance between two points on the Earth given their latitudes and longitudes.
        
        Parameters:
            self (object): The starting point with latitude and longitude coordinates.
            other (object): The ending point with latitude and longitude coordinates.
        
        Returns:
            float: The distance between the two points in kilometers.
        """
        distance_lat = radians(other.latitude - self.latitude)
        distance_lon = radians(other.longitude - self.longitude)
        a = sin(distance_lat / 2) ** 2 + cos(self.latitude) * cos(other.latitude) * sin(distance_lon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return self.R_TERRE * c

    def __str__(self):
        return f'Lieu {self.id}'


class Client(Lieu):
    def __init__(self, id: int, longitude: float, latitude: float, demande: float):
        """
        Initialize the object with the provided id, longitude, latitude, and demande.

        Parameters:
            id (int): The unique identifier for the object.
            longitude (float): The longitude coordinate of the object.
            latitude (float): The latitude coordinate of the object.
            demande (float): The demand associated with the object.
        """
        super().__init__(id, longitude, latitude)
        self.demande = demande

    def __str__(self):
        return f'{self.id}'

    def __repr__(self):
        return f'{self.id}'


class Depot(Lieu):
    def __init__(self, id: int, longitude: float, latitude: float):
        super().__init__(id, longitude, latitude)

    def __str__(self):
        return f'Depot {self.id}'


class Vehicule():
    def __init__(self, id: int, capacite: float):
        """
        Initialize the object with the provided id and capacite.

        Parameters:
            id (int): The unique identifier for the object.
            capacite (float): The capacity value assigned to the object.
        """
        self.id = id
        self.capacite = capacite

    def __str__(self):
        return str(self.id)

    def __repr__(self):
        return str(self.id)

    def mesure_load(self, actions: list[Lieu]) -> float:
        """
        Calculate the remaining load based on the actions taken.

        Parameters:
            actions (list[Lieu]): a list of actions taken, could be either Client or Depot objects.

        Returns:
            float: the remaining load after processing the actions.
        """
        load = self.capacite
        for action in actions:
            if isinstance(action, Client):
                load -= action.demande
            elif isinstance(action, Depot):
                load = self.capacite
        return load

    def mesure_distance(self, actions: list[Lieu]) -> float:
        """
        Calculate the total distance between the given list of locations.

        Parameters:
            actions (list[Lieu]): A list of locations to calculate the distance between.

        Returns:
            float: The total distance between all the locations in the list.
        """
        if len(actions) <= 1: return 0
        d = 0
        for i in range(len(actions) - 1):
            d += actions[i].distance(actions[i + 1])
        return d


# SOLUTIONS


class SolutionFinale():
    def __init__(self, schedule: dict[Vehicule, list[Lieu]], probleme : 'SMAVRP'):
        """
        Initialize the instance with the given schedule and problem.

        :param schedule: A dictionary mapping Vehicule objects to a list of Lieu objects.
        :param probleme: The problem to be initialized.
        """
        self.schedule = schedule
        self.probleme = probleme

    @property
    def nb_vehicles(self) -> int:
        return len(self.schedule)

    def impose_capacity_constraint(self):
        """
        Imposes trips back to the depot to the vehicle if it does not have enough materials to deliver the next customer.
        """
        for vehicule in self.schedule:
            n = len(self.schedule[vehicule])
            if n == 0:
                pass
            done = []
            to_do = self.schedule[vehicule]
            while to_do:
                next_stop = to_do.pop(0)
                if isinstance(next_stop, Client) and next_stop.demande > vehicule.mesure_load(done):
                    done.append(self.probleme.depot)
                done.append(next_stop)
            self.schedule[vehicule] = done

    def score(self) -> float:
        """
        Calculate the total score of the current schedule based on the vehicle distances.
        Return the total score as a float value.
        """
        self.impose_capacity_constraint()
        for vehicule in self.schedule:
            if len(self.schedule[vehicule]) > 0 and not isinstance(self.schedule[vehicule][-1], Depot):
                self.schedule[vehicule].append(self.probleme.depot)
            if len(self.schedule[vehicule]) > 0 and not isinstance(self.schedule[vehicule][0], Depot):
                self.schedule[vehicule].insert(0, self.probleme.depot)
        c = self.nb_vehicles * self.probleme.cout_vehicule
        distances = [vehicule.mesure_distance(self.schedule[vehicule]) for vehicule in self.schedule]
        c += sum(distances) + max(distances) * self.nb_vehicles
        return c

    def score_2(self) -> float:
        """
        A function to calculate the total cost of a schedule including vehicle constraints.
        """
        self.impose_capacity_constraint()
        total_cost = self.nb_vehicles * self.probleme.cout_vehicule

        for vehicle, path in self.schedule.items():
            # Créer une copie temporaire de l'itinéraire pour le véhicule
            temp_path = [self.probleme.depot] + path[:] if not isinstance(path[0], Depot) else path[:]
            if len(path) > 0 and not isinstance(path[-1], Depot):
                temp_path.append(self.probleme.depot)

            # Calculer la distance pour le chemin temporaire
            distance = vehicle.mesure_distance(temp_path)
            total_cost += distance

        return total_cost

    def __str__(self):
        return str(self.schedule)

    def plot(self):
        """
        A method to plot the schedule of vehicles on a map and vizualize the results of the computation.
        """
        self.impose_capacity_constraint()
        print(self.nb_vehicles)
        print()
        for vehicule in self.schedule:
            if len(self.schedule[vehicule]) > 1 and not isinstance(self.schedule[vehicule][-1], Depot):
                self.schedule[vehicule].append(self.probleme.depot)
            if len(self.schedule[vehicule]) > 1 and not isinstance(self.schedule[vehicule][0], Depot):
                self.schedule[vehicule].insert(0, self.probleme.depot)
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        imagery = OSM()
        ax.add_image(imagery, 9)
        for vehicule in self.schedule:
            if len(self.schedule[vehicule]) > 1:
                ax.plot([l.longitude for l in self.schedule[vehicule]], [l.latitude for l in self.schedule[vehicule]])
                #for l in self.schedule[vehicule] :
                #plt.text(l.longitude , l.latitude, str(l.id), horizontalalignment='center',verticalalignment='center')
        ax.plot(self.probleme.depot.longitude, self.probleme.depot.latitude, 'ro')
        plt.figure()
        i = 0
        for vehicule in self.schedule:
            if len(self.schedule[vehicule]) > 1:
                origin = self.schedule[vehicule][0]
                time = 0
                #plt.text(0,len(self.schedule)-i, self.schedule[vehicule][0], horizontalalignment='center',verticalalignment='center')
                for j in range(1, len(self.schedule[vehicule])):
                    plt.plot([time, time + origin.distance(self.schedule[vehicule][j])],
                             [len(self.schedule) - i - j / len(self.schedule[vehicule]),
                              len(self.schedule) - i - j / len(self.schedule[vehicule])])
                    #plt.text(time + origin.distance(self.schedule[vehicule][j]), len(self.schedule)-i-(j+0.5)/len(self.schedule[vehicule]), self.schedule[vehicule][j], horizontalalignment='center',verticalalignment='center')
                    time += origin.distance(self.schedule[vehicule][j])
                    origin = self.schedule[vehicule][j]
                i += 1
        plt.figure()


class Solution():
    def __init__(self, chromosome: list[Client], n_vehicules: int, probleme : 'SMAVRP'):
        """
        Initializes the chromosome, number of vehicles, and problem for the genetic algorithm.

        Parameters:
            chromosome (list[Client]): The chromosome representing the solution.
            n_vehicules (int): The number of vehicles available.
            probleme ('SMAVRP'): The SMAVRP problem instance.
        """
        self.chromosome = chromosome
        self.N_vehicules = n_vehicules
        self.probleme = probleme

    def __str__(self):
        return f"{[str(l) for l in self.chromosome]} by {self.N_vehicules} vehicles"

    def generate_neighbors(self, num_neighbors: int = 10) -> list[tuple['Solution', int]]:

        """
        Function that generates direct neighbors from the solution
        
        Parameters:
            num_neighbors (int): Number of neighbor to generate
            
        Returns:
            neighbors (list[Client]): List of random neighbors
        """

        return [self.Qstep() for _ in range(num_neighbors)]

    def to_int_id(self):
        """
        A function to convert the chromosome to an unique integer ID.
        """
        n = len(self.chromosome)
        result = 0
        for i in range(n):
            result += (self.chromosome[i].id - 1) * factorial(n - i - 1)
        return result + 1

    def elementary_move(self, i1: int, i2: int, n: int):
        """
        Perform an elementary move on the chromosome by swapping the elements at positions i1 and i2. 
        Update the number of vehicles to be the maximum of 1 and the current number of vehicles plus n. 
        Return a new Solution object with the updated chromosome, vehicle count, and problem.
        """
        chromosome = self.chromosome
        N = self.N_vehicules
        chromosome[i1], chromosome[i2] = chromosome[i2], chromosome[i1]
        N = max(1, N + n)
        return Solution(chromosome, N, self.probleme)

    def mutate(self, p: float) -> tuple['Solution', str]:
        """
        Generates a random mutation of the solution, and returns the new solution and the move.
        """
        i1 = 0
        i2 = 0
        n = 0
        if rd.random() > p:
            i1 = rd.randint(0, len(self.chromosome) - 2)
            i2 = rd.randint(i1+1, len(self.chromosome) - 1)
        if rd.random() > p:
            n = rd.randint(-1, 1)
        N = len(self.chromosome)
        move = (i1 <<40)|(i2<<20)|(n+1)
        return self.elementary_move(i1, i2, n), move

    def move(self, eps):
        """
        A function to decide and apply the next move based on epsilon-greedy strategy.
        
        Parameters:
            eps (float): The epsilon value for the epsilon-greedy strategy.
            
        Returns:
            tuple: A tuple containing the next move to be executed and the move itself.
        """
        int_id = self.to_int_id()
        if rd.random() > eps and int_id in self.probleme.Qgrid.keys() :
            move: int = max(self.probleme.Qgrid[int_id], key=lambda k: self.probleme.Qgrid[int_id][k])
            i1 = (move>>40) & 0xFFFFF
            i2 = (move>>20) & 0xFFFFF
            n = (move & 0xFFFFF) -1
            return self.elementary_move(i1, i2, n), move
        else:
            sol, move = self.mutate(self.probleme.Pmut)
            return sol, move

    def Qstep(self) -> tuple['Solution', int]:
        """
        A function that performs the Q-learning step. Updates the Q-grid based on the current state, action, and rewards.
        Returns the next solution that has been determined by epsilon-greedy and the encoding for the move leading to it.
        """
        sol_suivante, move = self.move(0.8)
        recompense = 1 / sol_suivante.score()
        sol_optimale_future, move_optimal_futur = sol_suivante.move(0)
        actual_id = self.to_int_id()
        id_suivant = sol_suivante.to_int_id()
        if actual_id not in self.probleme.Qgrid.keys():
            self.probleme.Qgrid[actual_id] = {move: 0}
        if move not in self.probleme.Qgrid[actual_id].keys():
            self.probleme.Qgrid[actual_id][move] = 0
        if id_suivant not in self.probleme.Qgrid.keys():
            self.probleme.Qgrid[id_suivant] = {move_optimal_futur: 0}
        if move_optimal_futur not in self.probleme.Qgrid[id_suivant].keys():
            self.probleme.Qgrid[id_suivant][move_optimal_futur] = 0

        self.probleme.Qgrid[actual_id][move] += 0.1 * (
                    recompense + 0.9 * self.probleme.Qgrid[id_suivant][move_optimal_futur] -
                    self.probleme.Qgrid[actual_id][move])
        return sol_suivante, move

    def distance_genetique(self, other : 'Solution'):
        """
        Calculate the genetic distance between two solutions.

        Parameters:
        - other: Solution object, the other solution to compare to.

        Returns:
        - int, the genetic distance between the two solutions.
        """
        list1 = self.chromosome
        list2 = other.chromosome
        lcs_matrix = [[0 for _ in range(len(list2) + 1)] for _ in range(len(list1) + 1)]
        for i in range(len(list1) - 1, -1, -1):
                for j in range(len(list2) - 1, -1, -1):
                    if list1[i] == list2[j]:
                        lcs_matrix[i][j] = 1 + lcs_matrix[i + 1][j + 1]
                    else:
                        lcs_matrix[i][j] = max(lcs_matrix[i + 1][j], lcs_matrix[i][j + 1])
        return len(list1) - lcs_matrix[0][0]    


    def cross(self, other: 'Solution', p):
        """
        Generate new solutions by crossing chromosomes from two solutions if a random number is greater than breeding probablity.
        
        Parameters:
            other (Solution): Another solution to cross chromosomes with.
            p: A breeding probability threshold.
        
        Returns:
            list[Solution]: A list of new solutions created by crossing chromosomes, or the original solutions if the random number is not greater than p.
        """
        if self.probleme != other.probleme:
            print(f'p1 : {self.probleme}, {other.probleme}')
            raise ValueError("The two solutions must be from the same problem")
        if rd.random() > p:
            a = rd.randint(0, len(self.chromosome))
            dummy = Client(0, 0, 0, 0)
            fils1: list[Client] = [dummy] * len(self.chromosome)
            fils2: list[Client] = [dummy] * len(self.chromosome)
            fils3: list[Client] = [dummy] * len(self.chromosome)
            fils4: list[Client] = [dummy] * len(self.chromosome)
            fils1[:a] = self.chromosome[:a]
            fils2[:a] = other.chromosome[:a]
            fils3[a:] = self.chromosome[a:]
            fils4[a:] = other.chromosome[a:]
            i = 0
            for t in other.chromosome:
                if t not in fils1:
                    fils1[a + i] = t
                    i += 1
            i = 0
            for t in self.chromosome:
                if t not in fils2:
                    fils2[a + i] = t
                    i += 1
            i = 0
            for t in other.chromosome:
                if t not in fils3:
                    fils3[i] = t
                    i += 1
            i = 0
            for t in self.chromosome:
                if t not in fils4:
                    fils4[i] = t
                    i += 1

            return [Solution(fils1, self.N_vehicules, self.probleme), Solution(fils2, other.N_vehicules, self.probleme),
                    Solution(fils3, self.N_vehicules, self.probleme), Solution(fils4, other.N_vehicules, self.probleme)]
        else:
            return [self, other]

    @property
    def final(self):
        """
        This function generates the final solution by creating a list of vehicles, initializing a dictionary with empty lists for each vehicle, assigning tasks to vehicles based on the chromosome, removing empty lists, and returning the final solution object.
        """
        vehicules = [Vehicule(i, self.probleme.capacite_vehicule) for i in range(self.N_vehicules)]
        res = {v: [] for v in vehicules}
        for i, c in enumerate(self.chromosome):
            res[vehicules[i % self.N_vehicules]].append(c)
        res = {key: res[key] for key in res if len(res[key]) > 0}
        return SolutionFinale(res, self.probleme)

    def score(self) -> float:
        """
        Calculate and return the score of the object.
        Returns:
            float: The score of the object.
        """
        return self.final.score()


# MODELE

class SMAVRP(Model):
    def __init__(self, clients: list[Client], depot: Depot, cout_vehicule: float, capacite_vehicule: float,
                 pmut: float):
        """
        Initialize the model with the given parameters.

        Parameters:
            clients (list[Client]): List of customers
            depot (Depot): The depot location
            cout_vehicule (float): Cost per vehicle
            capacite_vehicule (float): Capacity of each vehicle
            pmut (float): Mutation probability

        Returns:
            None
        """
        super().__init__()
        self.clients = clients
        self.Pmut = pmut
        self.cout_vehicule = cout_vehicule
        self.depot = depot
        self.capacite_vehicule = capacite_vehicule
        self.best_solution = self.gen_random_solution(3)
        self.good_solution_pool: list[Solution] = []
        self.schedule = SimultaneousActivation(self)
        self.datacollector = DataCollector(
            model_reporters={"best_solution": lambda m: m.best_solution.score()},
            agent_reporters={"Score": lambda a: a.best_score}
        )

        self.Qgrid = {}

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

    def insert_solution_in_pool(self, solution: Solution):
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
        return

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()


# AGENTS

class MyAgent(Agent):
    def __init__(self, unique_id: int, model: SMAVRP, pop_size: int):
        """
        Initialize the algorithm with the given unique identifier, model, and population size.

        Parameters:
            unique_id (int): The unique identifier for the algorithm.
            model (SMAVRP): The model for the algorithm.
            pop_size (int): The size of the population.
        """
        super().__init__(unique_id, model)
        self.model = model
        self.population: list[Solution] = sorted(
            [self.model.gen_random_solution(rd.randint(2, 10)) for _ in range(pop_size)],
            key=lambda x: x.score()
        )
        self.best_score = self.population[0].score()

    def fetch_better_solutions_from_pool(self):
        """
        Fetches better solutions from the solution pool and updates the population accordingly.
        """
        to_get = [sol for sol in self.model.good_solution_pool if sol.score() < self.best_score]
        n = len(self.population)
        self.population = sorted(to_get + self.population, key=lambda x: x.score())
        self.population = self.population[:n]

    def push_solution_in_pool(self, sol: Solution):
        """
        A function to push a solution into the solution pool if it meets the genetic diversity criteria.

        Parameters:
            sol (Solution): The solution to be pushed into the pool.

        Returns:
            None
        """
        if not isinstance(sol, Solution):
            raise Exception('wtf')
        for s in self.model.good_solution_pool:
            if s.distance_genetique(sol) < len(sol.chromosome)//2:
                return
        self.model.insert_solution_in_pool(sol)

    def step(self):
        pass


class GeneticAgent(MyAgent):
    def __init__(self, unique_id: int, model: SMAVRP, pop_size: int, pcross: float):
        """
        Initialize the VRP agent with a unique ID, model, population size, crossover probability, and mutation probability.
        """
        super().__init__(unique_id, model, pop_size)
        self.Pcross = pcross
        self.probleme = model

    def step(self):
        """
        Generate the next generation of solutions based on the current population. 
        This function sorts the population, selects the top individuals, performs crossover and mutation, 
        updates the best score, and collects data for the model. 
        """
        self.fetch_better_solutions_from_pool()
        s: list[Solution] = sorted(self.population, key=lambda x: x.score())
        s: list[Solution] = s[:len(self.population)]
        nex_gen: list[Solution] = []
        for p1, p2 in zip(s[::2], s[1::2]):
            nex_gen.extend(p1.cross(p2, self.Pcross))
        nex_gen = sorted(nex_gen, key=lambda x: x.score())[:len(self.population)]
        self.population = [x.Qstep()[0] for x in nex_gen]
        self.population = sorted(self.population, key=lambda x: x.score())

        self.push_solution_in_pool(self.population[0])
        self.best_score = self.population[0].score()
        if self.model.best_solution.score() > self.best_score:
            self.model.best_solution = self.population[0]


class RSAgent(MyAgent):

    def __init__(self, unique_id: int, model: SMAVRP, pop_size: int, t0: float, cooling: float):

        """
        Create an instance of RSAgent

        Parameters:
            t0 (float): initial temperature
            cooling (float): cooling rate
        """

        super().__init__(unique_id, model, pop_size)
        self.t = t0
        self.cooling = cooling

    def step(self):

        """
        Makes the agent interact with the pool
        """

        self.fetch_better_solutions_from_pool()  #Update the population
        self.population: list[Solution] = sorted(self.population, key=lambda x: x.score())

        #Parameters for the algorithm
        s1: Solution = self.population[0]
        s2: Solution = s1
        t: float = self.t

        #Calculate the difference
        voisins = s2.generate_neighbors(1)

        s2 = voisins[0][0]
        f1: float = s1.score()
        f2: float = s2.score()
        df: float = f1 - f2

        if df < 0:
            s1 = Solution(s2.chromosome, s2.N_vehicules, s1.probleme)
        else:
            #Probability 
            prob: float = exp(-df / t)
            q: float = rd.random()
            if q < prob:
                s1 = Solution(s2.chromosome, s2.N_vehicules, s1.probleme)

        #Reduce the temperature
        t *= (1 - self.cooling)

        self.t = t

        #Update the attributes and the pool
        self.population.insert(0, Solution(s1.chromosome, s1.N_vehicules, s1.probleme))
        self.population.pop()
        self.best_score = self.population[0].score()
        self.push_solution_in_pool(self.population[0])

        if self.model.best_solution.score() > self.best_score:
            self.model.best_solution = self.population[0]



class TabouAgent(MyAgent):

    def __init__(self, unique_id: int, model: SMAVRP, pop_size: int, tabu_size: int = 10, neighbors: int = 10):
        super().__init__(unique_id, model, pop_size)
        self.nb_neighbors = neighbors
        self.tabou : list [int] = []
        self.size = tabu_size

    def step(self):
        """
        Update the population, generate better solutions, and update the best score accordingly.
        """

        self.fetch_better_solutions_from_pool()    ##Update the population
        self.population = sorted(self.population, key=lambda x: x.score())

        s1: Solution = self.population[0]
        s2: Solution = Solution(s1.chromosome, s1.N_vehicules, s1.probleme)

        neighbors = s2.generate_neighbors(self.nb_neighbors)
        neighbors = sorted(neighbors, key=lambda permutation: permutation[0].score())
        found = False
        for neighbor, move in neighbors:
            i1 = move >> 40 & 0xFFFFF
            i2 = move >> 20 & 0xFFFFF
            n = move & 0xFFFFF - 1
            reciprocal_move = (i1 << 40) | (i2 << 20) |(1-n)
            if move not in self.tabou and neighbor.score() < s1.score():
                s2 = neighbor
                s2_move = move
                s2_reciprocal_move = reciprocal_move
                if s2.score() < s1.score():
                    s1 = Solution(s2.chromosome, s2.N_vehicules, s2.probleme)
                found = True

        if found:
            self.tabou.append(s2_reciprocal_move)
            if len(self.tabou) > self.size:
                self.tabou.pop(0)

        self.population.insert(0, Solution(s1.chromosome, s1.N_vehicules, s1.probleme))
        self.population.pop()
        self.best_score = self.population[0].score()
        self.push_solution_in_pool(self.population[0])
        if self.model.best_solution.score() > self.best_score:
            self.model.best_solution = self.population[0]
