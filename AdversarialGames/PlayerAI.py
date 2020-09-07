from random import randint
from BaseAI import BaseAI
import numpy as np
import itertools
import math
import time
import sys

#sys.setrecursionlimit(2000)

class PlayerAI(BaseAI):
    def __init__(self):
        self.minUtility = 100000
        self.maxUtility = 0
        self.current_time = 0
        self.alpha = -1000000
        self.beta = 1000000
        self.minchild = None
        self.maxChild = None
        self.top_move = None
        self.copy_grid = None
        self.weights = [0.1, 0.9]
        self.tiles_possible = [2,4]
        self.probability = 0.9
        self.available_time = 0.06

    def getMove(self, grid):
        self.init_time = time.clock()
        self.copy_grid = grid.clone()
        utility_move = self.decision(grid)
        return utility_move
    
    def getNewTileValue(self):
        if randint(0,99) < 100 * self.probability:
            return self.tiles_possible[0]
        else:
            return self.tiles_possible[1]
    
    def substract_abs(self, x,y):
        return abs(x-y)    
    
    def monoticity(self, grid):
        counter = 0
        for i in range(grid.size):
            for j in range(grid.size):    
                value =  grid.map[i][j]
                
                if value == grid.map[i][min(j+1, grid.size-1)] or value == grid.map[min(i+1,grid.size-1)][j]:
                    counter = counter + abs(j - min(j+1, grid.size-1)) + abs(i - min(i+1, grid.size-1))
                    
                if value == grid.map[i][min(j-1, grid.size-1)] or value == grid.map[min(i-1,grid.size-1)][j]:
                    counter = counter + abs(j - max(j-1, grid.size-1)) + abs(i - max(i-1, grid.size-1))

        heuristic = 2*counter / (grid.size*grid.size)
        return heuristic

    def smoothness(self, grid):
        # Heuristix- Smoothness of tiles
        sum_diff = 0
        #column
        for i in range(grid.size - 1):
            sum_diff += sum(list(map(self.substract_abs, grid.map[i], grid.map[i+1])))
        #row 
        for i in range(grid.size):    
            sum_diff += sum(list(map(self.substract_abs, grid.map[i][:-1], grid.map[i][1:])))

        total_sum = sum(itertools.chain.from_iterable(grid.map))
        heuristic = sum_diff/(2*total_sum)

        return heuristic, total_sum

    def computeUtility(self, grid):
        # Heuristic 1 - available cells
        h1 = len(grid.getAvailableCells())/(grid.size*grid.size)

        # Smoothness 
        h2, total_sum = self.smoothness(grid)

        # Monoticity
        h3 = self.monoticity(grid)

        maxTile = grid.getMaxTile()

        h4 = maxTile/(total_sum)
        #print(str(h1) + " " + str(h2) + " " + str(h4))

        utility = h1 - 5*h2 - h4 + h3
        return utility
    
    def Minimize(self, grid, alpha, beta):
        
        # check if the time is nearly out to go out
        if self.terminalTest():
            return None, self.computeUtility(grid) #Compute utility

        (minChild, minUtility) = (None, np.inf)

        #moves = grid.getAvailableMoves() # Get possible moves
        cells = grid.getAvailableCells()

        for possible_cell in self.tiles_possible:
            for cell in cells:
                grid_child = grid.clone()
                #grid_child.move(move) # generate the child with the move
                grid_child.insertTile(cell, possible_cell)
                _, utility = self.maximize(grid_child, alpha, beta)

                if utility < minUtility:
                    minUtility = utility
                    minChild = cell

                if minUtility <= alpha:
                    break

                if minUtility < beta:
                    beta = minUtility
        return minChild, minUtility

    def maximize(self, grid, alpha, beta):
        
        # check if the time is nearly out to go out
        if self.terminalTest():
            return None, self.computeUtility(grid) #Compute utility

        (maxChild, maxUtility) = (None, -np.inf)
        moves = grid.getAvailableMoves() # Get possible moves

        for move in moves:
            grid_child = grid.clone()
            grid_child.move(move) # generate the child with the move
            _, utility = self.Minimize(grid_child, alpha, beta)

            if utility > maxUtility:
                maxUtility = utility
                maxChild = move
            
            if maxUtility >= beta:
                break

            if maxUtility > alpha:
                alpha = maxUtility

        return maxChild, maxUtility

    def terminalTest(self):
        if time.clock() - self.init_time > self.available_time:
            return True
        else:
            return False
    """
    def terminal_test(self, grid, depth):
        if depth >= MAXDEPTH or (not grid.canMove()):
            return True
        else:
            return False
    """
    def decision(self, grid):
        move, utility = self.maximize(grid, -np.inf, np.inf)
        #print(move)
        print(grid.map)
        return move

