import numpy as np #pour déclarer la grille en 2d
import time #comparaison entre dijkstra et A*
import matplotlib.pyplot as plt #l'affichage de la fenetre de la grille
import math #formule de math pour la diagonale
import random #Choisir aléatoirement les lieux de livraisons
import heapq #pour les fonctions de dijkstra
import sys #pour utiliser getsizeof

def ajoutmalus(grille, z):
    malus = np.zeros(grille.shape)
    for x1, y1, x2, y2, bonus in z:
        for i in range(min(x1, x2), max(x1, x2) + 1):
            for j in range(min(y1, y2), max(y1, y2) + 1):
                if 0 <= i < grille.shape[0] and 0 <= j < grille.shape[1]:
                    malus[i, j] = bonus
    return malus

# Le 0 c'est un obstacle et le 1 un chemin libre
grille = np.array([ #Je déclare le grid 
[0, 0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	1, 0, 1, 1,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 1,	1, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 1, 0,	0, 1, 1, 1,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 1,	0, 1, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	1, 1, 0, 0,	0, 0, 0, 0,	0, 0, 0, 1,	0, 0, 1, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 1,	0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 0,	0, 0, 0, 1,	1, 1, 0, 0,	0, 0, 1, 0,	0, 0, 1, 0,	1, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 1, 1,	1, 0, 1, 0,	0, 0, 1, 0,	0, 1, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 1, 0, 0,	0, 1, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 1, 0, 0,	1, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 1, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	1, 0, 1, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 1, 0, 1,	0, 0, 0, 0,	0, 0, 0, 0,	0, 1, 0, 1,	0, 0, 0, 1,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 1, 1, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 1, 1, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 1,	0, 0, 0, 0,	0, 1, 1, 0,	0, 0, 0, 0,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 1, 1,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 1, 0, 0,	1, 1, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 1,	0, 1, 0, 0,	0, 0, 1, 0,	1, 0, 0, 0,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 1, 1,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 1,	0, 0, 1, 0,	0, 1, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	1, 1, 0, 0,	0, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 1, 0,	0, 1, 0, 1,	1, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 1, 0, 0,	1, 0, 0, 0,	1, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 0,	1, 0, 1, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	0, 0, 0, 0,	0, 1, 1, 0,	1, 0, 0, 1,	0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 1,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 1,	1, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 1,	0, 0, 1, 0,	0, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	1, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	1, 1, 1, 1,	1, 1, 0, 0,	0, 0, 0, 0,	0, 0, 1, 0,	0, 1, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 1, 0,	1, 1, 1, 0,	0, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	1, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 1, 1,	1, 0, 0, 0,	0, 1, 0, 0,	0, 0, 1, 1,	1, 0, 1, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 1, 0,	0, 0, 1, 0,	0, 0, 0, 0,	0, 0, 0, 1,	1, 0, 0, 0,	0, 1, 0, 0],
[0, 0, 0, 0, 1,	1, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	0, 1, 1, 0,	1, 0, 0, 0,	0, 1, 0, 0,	0, 1, 1, 1,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 1,	0, 0, 0, 0,	0, 0, 1, 0,	0, 1, 0, 0,	0, 0, 1, 0],
[0, 0, 0, 1, 0,	0, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 1,	1, 0, 0, 0,	1, 0, 0, 0,	0, 0, 1, 0,	1, 1, 1, 0,	0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 1, 0, 0,	0, 0, 1, 0,	0, 0, 1, 0],
[0, 0, 0, 1, 0,	0, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 1, 1, 1,	0, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 1,	1, 1, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	0, 1, 0, 0,	1, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 1],
[0, 0, 0, 0, 1,	0, 0, 0, 0,	0, 0, 0, 0,	0, 1, 0, 1,	1, 1, 1, 1,	0, 0, 0, 0,	0, 0, 0, 1,	1, 1, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	0, 0, 1, 1,	1, 0, 0, 0,	0, 0, 0, 0,	0, 0, 1, 1,	0, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 1],
[0, 0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	1, 1, 0, 0,	0, 0, 0, 1,	0, 0, 1, 0,	0, 1, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	0, 0, 0, 1,	1, 0, 0, 1],
[0, 0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 0,	0, 0, 1, 1,	0, 0, 0, 0,	1, 0, 0, 1,	1, 1, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 1, 1,	0, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 1, 0,	1, 0, 0, 1],
[0, 0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 1,	1, 1, 0, 0,	1, 0, 0, 0,	0, 1, 0, 0,	0, 1, 1, 1,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	1, 1, 1, 0,	0, 0, 0, 0,	0, 1, 0, 0,	0, 1, 0, 0,	1, 0, 0, 1],
[0, 0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 1, 0,	0, 1, 0, 0,	1, 1, 1, 1,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 1,	1, 0, 0, 0,	0, 0, 1, 0,	1, 0, 0, 0,	1, 0, 0, 1],
[0, 0, 0, 0, 0,	0, 0, 1, 0,	0, 0, 1, 0,	0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 1,	0, 1, 0, 0,	0, 0, 0, 0,	1, 1, 1, 1,	0, 0, 0, 0,	0, 0, 0, 0,	0, 1, 1, 0,	0, 0, 0, 1,	0, 0, 0, 0,	1, 0, 0, 1],
[0, 0, 0, 0, 0,	0, 0, 0, 1,	0, 1, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	0, 0, 1, 0,	1, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 1,	1, 1, 0, 1,	1, 0, 0, 0,	1, 0, 1, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 1,	0, 1, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 1,	1, 0, 0, 0,	0, 0, 0, 0,	0, 1, 0, 0,	0, 1, 1, 1,	1, 0, 0, 0,	0, 0, 0, 0,	0, 0, 1, 0,	0, 1, 1, 1,	1, 1, 1, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 1,	0, 1, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 1, 0,	0, 0, 1, 0,	1, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	0, 1, 1, 1,	1, 1, 1, 0,	0, 0, 1, 0,	0, 0, 0, 0,	1, 0, 1, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 1, 0,	0, 1, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 1,	0, 0, 1, 0,	0, 0, 0, 1,	0, 0, 1, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 1,	1, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	1, 1, 1, 0,	0, 0, 0, 1,	0, 0, 1, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 0,	0, 1, 0, 0,	0, 1, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 1,	0, 0, 1, 0,	0, 0, 1, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 0,	0, 1, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	1, 1, 0, 0,	0, 0, 1, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 0,	0, 1, 0, 1,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	1, 0, 1, 0,	0, 1, 0, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 1,	1, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 0,	0, 1, 0, 1,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	1, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 1,	1, 1, 1, 1,	1, 1, 1, 1,	1, 1, 1, 1,	1, 1, 1, 1,	1, 1, 1, 1,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 1, 0,	1, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 1,	0, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	0, 1, 1, 1,	0, 0, 0, 0,	0, 1, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	0, 0, 0, 0,	0, 1, 1, 1,	0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	1, 0, 1, 1,	1, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 1, 0,	1, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 1, 1, 1,	1, 1, 1, 1,	1, 1, 1, 1,	1, 1, 0, 0,	0, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 1, 0, 0,	0, 1, 0, 1,	0, 0, 0, 0,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	0, 0, 0, 0],
[0, 0, 0, 0, 1,	0, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 1,	1, 1, 1, 1,	1, 1, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0],
[0, 0, 0, 1, 0,	0, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	0, 0, 1, 0,	0, 0, 1, 1,	1, 1, 1, 1,	1, 1, 1, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0],
[0, 0, 1, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	0, 0, 0, 1,	0, 1, 0, 0,	0, 0, 0, 0,	1, 0, 0, 1,	0, 0, 0, 0,	0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0],
[0, 1, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0],
[1, 0, 1, 0, 0,	0, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0],
[0, 0, 0, 1, 0,	0, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 0,	0, 0, 1, 0,	1, 1, 1, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0],
[0, 0, 0, 0, 1,	0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	1, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	1, 1, 1, 1,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 1, 0,	1, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 1, 0, 1,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 1,	1, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 1, 0,	0, 1, 0, 0,	0, 1, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 1, 0,	1, 1, 1, 0,	1, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 1, 0,	0, 0, 1, 0,	1, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 1, 0,	0, 0, 0, 1,	0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 1, 1, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 1,	0, 0, 1, 0,	0, 0, 0, 0,	1, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 1,	1, 1, 0, 0,	0, 0, 0, 0,	0, 1, 0, 0,	0, 0, 0, 0,	1, 1, 0, 0,	0, 0, 0, 1,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 1, 1,	1, 1, 1, 1,	1, 1, 1, 1,	1, 1, 1, 1,	1, 0, 0, 0,	0, 0, 1, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 1, 0, 0,	0, 1, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0],
[0, 0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 1, 1,	1, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0]
])
ligne, colonne = grille.shape

n = int(input("Nombre de livraisons : "))

#Fonction sliman de centre vill et travauxe event
if input("Centre ville plein de monde? (oui ou non) : ") == "oui":
    centreville = [(68, 50, 37, 33, float(input("Coefficient de malus : ")))]
else:
    centreville = [(0, 0, 0, 0, 0.0)]

malus = ajoutmalus(grille, centreville)
n_event = int(input("Combien de travaux/évenements à générer?: "))
for i in range(n_event):
    coordonnee_travaux = input(f"rentrer les coordonné des travaux/events {i+1} (x,y): ")
    x, y = map(int, coordonnee_travaux.split(','))
    if 0 <= x < ligne and 0 <= y < colonne:
        grille[x, y] = 0  # Ligne ajoutée

#Fonction de Haemmeryck
n_event = int(input("Combien de feux à créer?: "))
feux = []
for i in range(n_event):
    # Position du feu
    coordonnee_feu = input(f"Position feu {i+1} (x,y): ")
    x, y = map(int, coordonnee_feu.split(','))
    bonus_feu = float(input(f"Temps feu rouge {i+1}: "))
    malus[x, y] += bonus_feu  # Toujours malus pour le calcul
    feux.append((x, y))

#Déplacement en diagonale 
def deplacement_diagonal(position, malus):
    x, y = position
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]: #gauche droite haut bas diagonal haut gauche haut droit bas gauche bas droit
        nx, ny = x + dx, y + dy
        if 0 <= nx < ligne and 0 <= ny < colonne and grille[nx][ny] == 1:
            cout = (1.0 if dx == 0 or dy == 0 else math.sqrt(2)) + malus[nx, ny] #hypoténuse pour déplacement en diagonale
            yield (nx, ny), cout

def heuristique(a, b, type="euclidien"):
    if type == "manhattan":
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def dijkstra(depot, objectif, malus):
    open_set = [(0, depot)]
    noeud = {depot: {'sommes': 0, 'reconstruction': None}}
    exploré = 0

    while open_set:
        _, noeud_actuel = heapq.heappop(open_set)
        exploré += 1

        if noeud_actuel == objectif:
            chemin = []
            while noeud_actuel is not None:
                chemin.append(noeud_actuel)
                noeud_actuel = noeud[noeud_actuel]['reconstruction']
            return chemin[::-1], noeud[chemin[0]]['sommes'], exploré

        for voisin, cout in deplacement_diagonal(noeud_actuel, malus):
            nouveau_cout = noeud[noeud_actuel]['sommes'] + cout
            if voisin not in noeud or nouveau_cout < noeud[voisin]['sommes']:
                noeud[voisin] = {'sommes': nouveau_cout, 'reconstruction': noeud_actuel}
                heapq.heappush(open_set, (nouveau_cout, voisin))

    return exploré

def astar(depot, objectif, malus, typeheuristique="euclidien"):
    open_set = [(0, depot)]
    noeud = {depot: {'sommes': 0, 'reconstruction': None}}
    exploré = 0

    while open_set:
        _, noeud_actuel = heapq.heappop(open_set)
        exploré += 1

        if noeud_actuel == objectif:
            chemin = []
            while noeud_actuel is not None:
                chemin.append(noeud_actuel)
                noeud_actuel = noeud[noeud_actuel]['reconstruction']
            return chemin[::-1], noeud[chemin[0]]['sommes'], exploré

        for voisin, cout in deplacement_diagonal(noeud_actuel, malus):
            nouveau_cout = noeud[noeud_actuel]['sommes'] + cout
            if voisin not in noeud or nouveau_cout < noeud[voisin]['sommes']:
                noeud[voisin] = {'sommes': nouveau_cout, 'reconstruction': noeud_actuel}
                f_score = nouveau_cout + heuristique(voisin, objectif, typeheuristique)
                heapq.heappush(open_set, (f_score, voisin))

    return [], float('inf'), exploré

def glouton(depot, destinations, malus, algo="astar", typeheuristique="euclidien"): #glouton utiliser avec l'algoritme de astar en euclidien
    noeud_actuel = depot
    ordre = []
    chemintotal = []
    cout_total = 0
    exploré_total = 0
    pasvisiter = destinations.copy()

    while pasvisiter:
        meilleur = None
        meilleur_cout = float('inf')
        meilleur_chemin = []
        exploré_meilleur = 0

        for d in pasvisiter:
            if algo == "dijkstra":
                chemin, cout, exploré = dijkstra(noeud_actuel, d, malus)
            else:
                chemin, cout, exploré = astar(noeud_actuel, d, malus, typeheuristique)
            if chemin and cout < meilleur_cout:
                meilleur = d
                meilleur_cout = cout
                meilleur_chemin = chemin
                exploré_meilleur = exploré

        if meilleur:
            pasvisiter.remove(meilleur)
            ordre.append(meilleur)
            chemintotal.extend(meilleur_chemin[:-1])
            cout_total += meilleur_cout
            exploré_total += exploré_meilleur
            noeud_actuel = meilleur

    if ordre:
        if algo == "dijkstra":
            chemin, cout, exploré = dijkstra(noeud_actuel, depot, malus)
        else:
            chemin, cout, exploré = astar(noeud_actuel, depot, malus, typeheuristique)
        if chemin:
            chemintotal.extend(chemin)
            cout_total += cout
            exploré_total += exploré

    return ordre, chemintotal, cout_total, exploré_total

def compare_algorithms(depot, destinations, malus):
    resultats = {}
    algorithmes = [ # dictionnaire de mots utiliser pour affichage du prompt
        ("dijkstra", "dijkstra", None), 
        ("astar euclidienne", "astar", "euclidien"),
        ("astar de manhattan", "astar", "manhattan")
    ]

    for nom, algo, heur in algorithmes:
        t0 = time.time()
        ordre, trajet, cout, exploré = glouton(depot, destinations.copy(), malus, algo, heur)
        t1 = time.time()
        resultats[nom] = {
            "ordre": ordre,
            "trajet": trajet,
            "longueur": cout,
            "temps": t1 - t0,
            "explorés": exploré
        }

    return resultats

def afficher_comparaison(resultats): #infos afficher pour chaque types d'algo
    for algo, r in resultats.items():
        print(f"\n{algo.upper()}")
        print(f"Ordre de visite: {r['ordre']}")
        print(f"Longueur totale: {r['longueur']:.2f}")
        print(f"Nœuds explorés: {r['explorés']}")
        print(f"Temps d'exécution: {r['temps']:.5f} sec\n")

cellulechemin = [(i,j) for i in range(ligne) for j in range(colonne) if grille[i][j] == 1]
depot = (55, 0)

destinations = random.sample([cell for cell in cellulechemin if cell != depot], min(n, len(cellulechemin)-1))
ordre, trajet, cout, exploré = glouton(depot, destinations, malus, "astar", "euclidien")

resultats = compare_algorithms(depot, destinations, malus)
afficher_comparaison(resultats)

ordre, trajet, cout, exploré = glouton(depot, destinations, malus, "astar", "euclidien")

fig, ax = plt.subplots(figsize=(12, 10))
ax.imshow(grille, cmap='gray')
ax.add_patch(plt.Rectangle((y - 0.5, x - 0.5), 1, 1, color='blue'))

for x1, y1, x2, y2, bonus in centreville: #modélisation du centre ville 
    ax.add_patch(plt.Rectangle((min(y1, y2) - 0.5, min(x1, x2) - 0.5), 
    abs(y2 - y1) + 1, abs(x2 - x1) + 1, 
    color='orange', alpha=0.3))

for x, y in trajet:
    ax.add_patch(plt.Rectangle((y - 0.5, x - 0.5), 1, 1, color='yellow', alpha=0.6))
    ax.add_patch(plt.Rectangle((depot[1] - 0.5, depot[0] - 0.5), 1, 1, color='green'))#modélisation du dépot sur la carte

for i, dest in enumerate(ordre):
    ax.add_patch(plt.Rectangle((dest[1] - 0.5, dest[0] - 0.5), 1, 1, color='red'))
    ax.text(dest[1], dest[0], str(i + 1), ha='center', va='center',
            fontweight='bold', fontsize=8, color='white')

print("\nRésumé du parcours :")#affichage dans le prompt des infos
print(f"Depot: {depot}")
print(f"Ordre de visite: {ordre}")
print(f"Cout total : {cout:.2f} cases")
print(f"Nœuds explorés : {exploré}")
print("Taille d'un 0 (en octets):", sys.getsizeof(0))
print("Taille d'un 1 (en octets):", sys.getsizeof(1))
plt.show()