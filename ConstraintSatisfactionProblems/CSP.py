
import sys
import time
import copy
import numpy as np
import queue as Q
import pandas as pd
import itertools
from string import ascii_letters

class csp(object):

    def __init__(self, sudoku, sudoku_board):
        self.sudoku = sudoku
        self.board = sudoku_board
        self.assigned = []

        for variable in self.sudoku:
            if self.sudoku[variable].value != 0:
                self.assigned.append(variable)

        print("Variables Assigned : " + str(len(self.assigned)))

    def Backtracking_search(self, constraints):
        """
        Returns a solution, or failure
        """
        return self.Recursive_BT_search(constraints)
    
    def Recursive_BT_search(self, constraints):
        """
        Returns a solution, or failure
        """
        # check if the assignment is done
        if len(self.assigned)==81:
            return True # return the complete variables assigned

        # Select the variable using Minimum Remaining Values
        variable = self.Select_Unassigned_Var()

        # Select the order domain to examine the variables
        #print("Variable : " + str(variable))
        for value in self.Order_Domain_Values(variable):
            #print("Value : " + str(value))
            # get the possitions constraints affected by the variable
            pos_constraints = [cons for cons in constraints if cons[0]==variable]
            if self.consistancy(value, pos_constraints):
                # Assigne value to the variable and the list of assignments
                self.assigned.append(variable)
                self.sudoku[variable].value = value
                # Forward checking
                inference = self.forward_checking(value, pos_constraints)
                if inference:
                    # again call the recursive function to continue the variable domain prunning
                    result = self.Recursive_BT_search(constraints)
                    # check if the backtracking search worked
                    if result:
                        return result
                    else:
                        # Failed the backtracking with this value, recover configuration
                        self.assigned.remove(variable)
                        self.sudoku[variable].value = 0

        # if noone of the possible values for variable worked return false meaning failure
        return False
    
    def BT_assignment(self):
        """
        Function to return the solution of the sudoku found by backtracking algorithm
        """
        result = []
        for variable in self.sudoku:
            result.append(self.sudoku[variable].value)
        return result
    
    def forward_checking(self, value, pos_constraints):
        """
        Function which delete value of variable in possible variables affected

        Input : value - integer to be deleted in variable domains
                pos_constraints - tuple of constraints affected by the variable (variable, var_affected)
                reverse - flag to process forward or backward the value in the variables affected
        """
        for cons in pos_constraints:
            if self.assigned.count(cons[1])==0: # Check unassigned variables
                # remove value from cons[1] domain
                self.sudoku[cons[1]].add_domain_value(value)
                if len(self.sudoku[cons[1]].domain) == 0:
                    return False
        return True

    def consistancy(self, value, pos_constraints):
        """
        Function that check the consistance of the value with the constraints

        Input : value - value to check
                pos_constraints - constraints interwined with the value variable

        return : True - if the value is consistant, False otherwise
        """
        # Check if the val choosen interwine with neighbors domains
        for pos_con in pos_constraints:
            if value == self.sudoku[pos_con[1]].value:
                return False
        return True

    def consistancy_val(self, value, pos_constraints):
        """
        Function that check the consistance of the value with the constraints

        Input : value - value to check
                pos_constraints - constraints interwined with the value variable

        return : True - if the value is consistant, False otherwise
        """
        consistant = []
        # Check if the val choosen interwine with neighbors domains
        for pos_con in pos_constraints:
            domain = self.sudoku[pos_con[1]].domain
            # if the domain have more than 1 value diff to value
            #is consistant
            if any(y!=value for y in domain):
                consistant.append(pos_con[1])
        #print(len(consistant))
        #print(len(pos_constraints))       
        if len(consistant) == len(pos_constraints):
            # means that all the constraints were OK with value
            return True
        else:
            return False

    def Order_Domain_Values(self, variable):
        """
        Function that decide the order which values will be examined

        input : variable - Object containing domain of it

        Return : list of values
        """
        # Get the neighbors of the variable
        neigh_vars = self.neighbors(variable)
        neighbors = []

        # Get the domains of the neihbors
        for neigh in neigh_vars:
            neighbors.append(self.sudoku[neigh].domain)

        # Build the order list vector
        order_vals = []
        #print(self.sudoku[variable].domain)
        for val in self.sudoku[variable].domain:
            Neigh_block = copy.deepcopy(neighbors)
            domains = []
            
            for idx, neigh in enumerate(Neigh_block):
                if neigh.count(val)==1:
                    neigh.remove(val)
                    
                domains.append(len(Neigh_block[idx]))
            flexibility = [len(domain) for domain in Neigh_block]
            order_vals.append((sum(flexibility), val))

        # order the heuristic in descending order of flexbility
        order_vals.sort(key=lambda tup: tup[0], reverse=False)
        #print(order_vals)
        domain_order = [tup[1] for tup in order_vals]
        return domain_order


    
    def Select_Unassigned_Var(self):
        """
        Select the next variable in the order given by
        heuristic Minimum remaining values

        Inputs : self.sudoku - Variables of the sudoku grid 
                self.assigned - list of variables already processed

        Return : Object variable
        """
        # Iterate over the dict
        possible_vars = []

        # Finde the minimum domain varibles
        for var in self.sudoku:
            lenght_vals = len(self.sudoku[var].domain)
            possible_vars.append((lenght_vals, var))

        # Sort elements by domain lenght
        possible_vars.sort(key=lambda tup: tup[0], reverse=True)

        # Find the variables hadn't been processed 
        for pos_value in possible_vars:
            if self.assigned.count(pos_value[1]) == 0:
                variable = pos_value[1]
                break
        
        #print("The next variable to handle is : " + variable)
        return variable

    def AC_3(self, constraints):
        """
        Inputs : CSP, a binary CSP with components (X, D, C)
        local variables : queue, a queue of arcs, initially the arcs in csp

        Returns false if an inconsistency is found and true otherwise
        """
        queue_list = Q.Queue()

        for cons in constraints:
            queue_list.put(cons)
        
        while not queue_list.empty():
            Xi, Xj = queue_list.get()
            Xi_domain = self.sudoku[Xi].domain
            if self.revise(constraints, Xi, Xj):
                if len(Xi_domain) == 0:
                    return False
                for Xk in self.neighbors(Xi):
                    #if Xk != Xj:
                    queue_list.put((Xk, Xi))
        return True
                
    def revise(self, constraints, Xi, Xj):
        """
        Returns true if we revise the domain of Xi
        """
        Xi_domain = (self.sudoku[Xi].domain)
        Xj_domain = (self.sudoku[Xi].domain)
        revised = False
        for x in Xi_domain:
            if any (y==x for y in Xj_domain):
                    self.sudoku[Xi].delete_domain_value(x)
                    revised = True
        return revised
    
    def neighbors(self, Xi):
        pos = np.where(self.board == Xi)
        
        neighbors_list = []
        neighbors_list.append((pos[0][0] + 1, pos[1][0]))
        neighbors_list.append((pos[0][0] - 1, pos[1][0]))
        neighbors_list.append((pos[0][0], pos[1][0] + 1))
        neighbors_list.append((pos[0][0], pos[1][0] - 1))

        lista = []
        for neigh in neighbors_list:
            if neigh[0] >= 0 and neigh[0] <= 8 and neigh[1] >= 0 and neigh[1] <= 8:
                lista.append(self.board[neigh[0],neigh[1]])

        return lista

    def assignment(self):
        solved_sudoku = []
        for key in self.sudoku:
            solved_sudoku.append(self.sudoku[key].domain)
        return solved_sudoku
