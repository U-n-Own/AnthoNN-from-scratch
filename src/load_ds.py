""" 
Info taken from paper : The MONK's Problems: A Comparative Study of Learning Algorithms

Here we load and process the dataset MONK, this dataset is taken from the UCI Machine Learning Repository.
It's used to measure performance on different learning algorithms on binary classification problems.

MONK's 3 problems realy on the artificial robot domain, in which the robots are described by 6 attributes:
- head_shape \in {square, round, octagon}
- body_shape \in {square, round, octagon}
- is_smiling \in {yes,no}
- holding \in {sword, balloon, flag}
- jacket_color \in {red, yellow, blue, green}
- has_tie \in {yes, no}

eg. of problem 1 is (head_shape = body_shape) or (jacket_color = red) 

Attribute information:
1. class: 0, 1
2. a1: 1, 2, 3
3. a2: 1, 2, 3
4. a3: 1, 2
5. a4: 1, 2, 3
6. a5: 1, 2, 3, 4
7. a6: 1, 2
8. Id: (A unique symbol for each instance)

# Training set ML-CUP22-TR.csv: id inputs target_x target_y (last 2 columns) 

In the csv we have the index of the sample, the inputs and the target values:
- inputs are 6 values  
- target_x and target_y are the last two columns 
- we want to predict a1 from a2..a7 
...

#TODO: One-hot encoding on the inputs
"""

import 
