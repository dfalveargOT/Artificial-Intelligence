"""
David Felipe Alvear Goyes
Columbia University
Artificial Intelligence Course EDX 
"""
import sys
import time
import copy
import numpy as np
import queue as Q
import pandas as pd
import itertools
from CSP import csp
from string import ascii_letters

class variable(object):

    def __init__(self, value, row, col, domain = [1,2,3,4,5,6,7,8,9]):
        self.value = int(value)
        if value != '0':
            self.domain = [int(value)]
        else:
            self.domain = domain
        self.row = row
        self.col = col

    def delete_domain_value(self, value):
        self.domain.remove(value)

class data_utils(object):

    def __init__(self, input, output):
        self.sudoku_input = input
        self.output = output
    
    def generate_dict(self):  
        """
            load_data - Read csv file with the linear dataset 
            - input_string : name of the csv file to read
            - return array with the data loaded
        """
        ascci_ = ascii_letters.upper()
        sudoku_board = np.zeros((9,9), dtype=object)
        sudoku = dict()
        counter = 1 # Columns manage
        letters = 0 # Rows manage

        for value in self.sudoku_input:
            posicion_string = "".join(ascci_[letters] + str(counter))
            variable_ = variable(value, ascci_[letters], counter)
            # Add value to the dictionary
            sudoku.update({posicion_string: variable_})
            # Create board
            sudoku_board[letters][counter - 1] = posicion_string
            # update control variables
            if counter == 9:
                counter = 0
                letters += 1
            counter += 1
        #print(sudoku_board[:,:,0])
        return sudoku, sudoku_board

    def constraints_generation(self, board):
        constraints = []
        # First Constraint - constraint along row

        for row in board:
            constraints += (list(itertools.permutations(row,2)))
        # Second Constraints - constraint along cols
        for cols in np.transpose(board):
            constraints += (list(itertools.permutations(cols,2)))

        # Third Constraint - Blocks 3X3 constraints
        #Constraints within each 3x3 square
        for row in [0,3,6]:
            for col in [0,3,6]:
                box = [val for row in board[row:row+3] for val in row[col:col+3]]
                constraints += list(itertools.permutations(box,2))
        constraints = list(set(constraints)) # delete duplicated constraints
        # Remove duplicated constraints
        print("Total constraints found : " + str(len(constraints)))
        return constraints
    
    def output_file(self, result, method):
        file = open("output.txt","w")
        file.write("{} {}".format(result, method))
        file.close()

def main(string_sudoku = "", output_name = ""):
    
    if string_sudoku == "":
        string_sudoku = "000260701680070090190004500820100040004602900050003028009300074040050036703018000"
        output_name = "output.txt"

    data_manager = data_utils(string_sudoku, output_name)
    sudoku, sudoku_board = data_manager.generate_dict()
    constraints = data_manager.constraints_generation(sudoku_board)

    ## Pass throught CSP solver
    ## AC-3 solver
    solved_AC3 = None
    solverCSP = csp(copy.deepcopy(sudoku), copy.deepcopy(sudoku_board))
    bool_ac3 = solverCSP.AC_3(constraints)
    print(bool_ac3)
    #if bool_ac3:  
    solved_AC3 = solverCSP.assignment()
    print(solved_AC3)
    
    return solved_AC3

main()