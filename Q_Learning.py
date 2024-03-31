#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
from random import randint
import random

class EnvGrid(object):
    """
        docstring forEnvGrid.
    """
    def __init__(self):
        super(EnvGrid, self).__init__()

        
        

        # Je définis ma grille & les récompenses
        self.grid = [
            [0, 0, 1],
            [0, -1, 0],
            [0, 0, 0]
        ]
        # Je définis la position initiale
        self.y = 2
        self.x = 0

        # Je définis les actions possibles

        self.actions = [
            [-1, 0], # Up
            [1, 0], #Down
            [0, -1], # Left
            [0, 1] # Right
        ]

    # Je définis une fonction qui reset l'environnement aux valeurs de base


    def reset(self):
        """
            Reset world
        """
        self.y = 2
        self.x = 0
        return (self.y*3+self.x+1)

    
    # Je définis une fonction pour se dépalcer dans l'environnement
    def step(self, action):
        """
            Action: 0, 1, 2, 3
        """
        self.y = max(0, min(self.y + self.actions[action][0],2))
        self.x = max(0, min(self.x + self.actions[action][1],2))
        return (self.y*3+self.x+1) , self.grid[self.y][self.x]

    
    # Je définis une fonction pour afficher l'environnement actuel
    def show(self):
        """
            Show the grid
        """
        print("---------------------")
        y = 0
        for line in self.grid:
            x = 0
            for pt in line:
                print("%s\t" % (pt if y != self.y or x != self.x else "X"), end="")
                x += 1
            y += 1
            print("")

    # Je définis une fonction pour vérifier si j'ai terminé dans l'environnement 
    # i.e. la voiture a trouvé la maison dans la grille

    def is_finished(self):
        return self.grid[self.y][self.x] == 1

def take_action(st, Q, eps):
    # Take an action
    if random.uniform(0, 1) < eps:
        #exploration
        action = randint(0, 3)
    else: # Or greedy action
        #exploitation
        action = np.argmax(Q[st])
    return action

if __name__ == '__main__':
    env = EnvGrid()
    état_actuel = env.reset()

    Q = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]

    for _ in range(100):
        # Reset the game
        #Je récupère l'état actuel
        état_actuel = env.reset()
    
        while not env.is_finished():
            #env.show()
            #demande quelles actions je vais effectuer 
            #at = int(input("$>"))
            
            
            #je choisis une action


            action = take_action(état_actuel, Q, 0.8)

            #je me déplace dans l'environnement en utilisant l'action que j'ai utilisée
            #retourne 2 valeurs : stp1 : le nouvel etat dans le quel je me retrouve & r: la récompense
            
            état_suivant, récompense = env.step(action)
            #print("s", stp1)
            #print("r", r)

            # Update Q function
            # chercher atp1, epsilone =0 pour etre sur que  je fais de l'exploitation pour trouver l'epérance max à t+1
            action_optimale_future = take_action(état_suivant, Q, 0.0)
            # learning rate=0.1 , gamma=0.9 (à quel point je donne de l'importance aux récompenses futures)
            Q[état_actuel][action] = Q[état_actuel][action] + 0.1*(récompense + 0.9*Q[état_suivant][action_optimale_future] - Q[état_actuel][action])
            # le prochain état sera l'état à t+1 et on recommence
            état_actuel = état_suivant
    for s in range(1, 10):
        print(s, Q[s])

