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

"""
import numpy
import pandas as pd
import numpy as np

def load_monks(directory, shuffle=False):
    """
    Restituisce il dataset MONK (suddiviso in target_input e target_output)
    sotto forma di matrice dopo aver applicato One-hot encoding all'input

    :param directory: directory del dataset
    :param shuffle: se True, fa il shuffle del dataset
    """

    monks_ds = pd.read_csv(directory, sep = ' ', header=None)
    monks_ds = monks_ds.drop(monks_ds.columns[0], axis=1) # La prima colonna contiene solo valori NaN
    monks_ds = monks_ds.drop(monks_ds.columns[-1], axis=1) # L'ultima colonna contiene l'id -> non ci serve

    if shuffle:
        monks_ds = monks_ds.sample(frac=1).reset_index(drop=True)

    target_input = monks_ds.drop(monks_ds.columns[0], axis=1)
    target_output = monks_ds.iloc[:, 0]

    # One-hot encode the categorical attributes
    target_input = pd.get_dummies(target_input, columns=[2,3,4,5,6,7])

    target_input = np.matrix(target_input)
    target_output = np.matrix(target_output).reshape(len(monks_ds), 1)

    return target_input, target_output

def load_cup_training(directory = '../datasets/CUP/ML-CUP22-TR.csv', shuffle=False):
    """
    Restituisce il dataset CUP sotto forma di matrice(suddiviso in target_input e target_output)

    :param directory: directory del dataset ML-CUP22-TR.csv
    :param shuffle: se True, fa il shuffle del dataset
    """

    cup_ds = pd.read_csv(directory, comment='#', header=None)
    cup_ds = cup_ds.drop(cup_ds.columns[0], axis=1) # prima colonna contiene l'id -> non ci serve

    if shuffle:
        cup_ds = cup_ds.sample(frac=1).reset_index(drop=True)

    target_input = cup_ds.iloc[:, :9]  # prende le prime 9 colonne
    target_output = cup_ds.iloc[:, -2:]  # prende le ultime 2 colonne

    target_input = np.matrix(target_input)
    target_output = np.matrix(target_output)

    return target_input, target_output

def load_matrix(directory):
    """
    Restitusce le matrice di input e output salvate in "directory"

    :param directory: directory dove sono salvate le matrici per il model_assessment e model_selection
    """

    target_inputs = np.load(f"{directory}/target_inputs.npy")
    target_inputs = np.matrix(target_inputs)

    target_outputs = np.load(f"{directory}/target_outputs.npy")
    target_outputs = np.matrix(target_outputs)

    return target_inputs, target_outputs


def data_set_partitioning(target_inputs: numpy.matrix, target_outputs: numpy.matrix, percentage_training_set: int) -> dict:
    """
    Partizione del dataset in model_selection set e validation set

    :param target_inputs: matrice contenente gli input
    :param target_outputs: matrice contenente gli output
    :param percentage_training_set: percentuale di dati da usare per il model_selection set
    :return: dizionario con i dati di model_selection e validation
    """

    len_training_set = (percentage_training_set*target_inputs.shape[0]) / 100
    len_training_set = round(len_training_set)
    len_validation_set = target_inputs.shape[0] - len_training_set

    target_inputs_training = target_inputs[:len_training_set, :]
    target_inputs_validation = target_inputs[-len_validation_set:, :]

    target_outputs_training = target_outputs[:len_training_set, :]
    target_outputs_validation = target_outputs[-len_validation_set:, :]

    assert target_inputs_training.shape[0] == target_outputs_training.shape[0]
    assert target_inputs_validation.shape[0] == target_outputs_validation.shape[0]

    assert target_inputs_training.shape[1] == target_inputs_validation.shape[1]
    assert target_outputs_training.shape[1] == target_outputs_validation.shape[1]

    assert target_inputs_training.shape[0] + target_inputs_validation.shape[0] == target_inputs.shape[0]
    assert target_outputs_training.shape[0] + target_outputs_validation.shape[0] == target_outputs.shape[0]

    return {
        'model_selection': {
            'inputs': target_inputs_training,
            'outputs': target_outputs_training
        },
        'validation': {
            'inputs': target_inputs_validation,
            'outputs': target_outputs_validation
        }
    }



