import os
import driver
import numpy

input_ = "sudokus_start.txt"
result_ = "sudokus_finish.txt"

# Read all the sudokus
input_file = open(input_)
in_sudokus = input_file.readlines()
result_file = open(result_)
ou_sudokus = result_file.readlines()

# Iterate over all the sudokus
for sudoku_in, sudoku_res in zip(in_sudokus, ou_sudokus):
    line_in = sudoku_in[:-1]
    line_ou = sudoku_res[:-1]

    # Pass throught the solver
    result = driver.main(line_in, "Out.txt")
    if result != line_ou:
        print("Sudoku solved ")
        print(result)
        