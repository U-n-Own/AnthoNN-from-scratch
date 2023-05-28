import numpy as np
from matplotlib import pyplot as plt


# Qui sotto ci sono le funzioni per fare il plot della varianza e del bias

def plot_training(training_error_list):
    mean_training = np.mean(training_error_list, axis=0)
    std_training = np.std(training_error_list, axis=0)

    #Training
    for training_error in training_error_list:
        training_plot, = plt.plot(training_error.tolist()[0], color = 'lightblue', label='training')
    mean_training_plot, = plt.plot(mean_training.tolist()[0], color = 'blue', label='mean on training')
    plt.fill_between(range(std_training.shape[1]),
                     np.array(mean_training-std_training)[0],
                     np.array(mean_training+std_training)[0],alpha=.1, color = 'lightblue')

    return training_plot, mean_training_plot

def plot_validation(validation_error_list):
    mean_validation = np.mean(validation_error_list, axis=0)
    std_validation = np.std(validation_error_list, axis=0)

    #Validation
    for validation_error in validation_error_list:
        validation_plot, = plt.plot(validation_error.tolist()[0], color = 'pink', label='validation')
    mean_validation_plot, = plt.plot(mean_validation.tolist()[0], color = 'red', label='mean on validation')

    plt.fill_between(range(std_validation.shape[1]),
                     np.array(mean_validation-std_validation)[0],
                     np.array(mean_validation+std_validation)[0],alpha=.1, color = 'pink')
    return validation_plot, mean_validation_plot


def plot_mean_and_std_validation(training_error_list, validation_error_list):
    mean_training = np.mean(training_error_list, axis=0)
    std_training = np.std(training_error_list, axis=0)
    mean_validation = np.mean(validation_error_list, axis=0)
    std_validation = np.std(validation_error_list, axis=0)

    plt.plot(mean_training.tolist()[0], color='blue', label='mean on training')
    plt.fill_between(range(std_training.shape[1]),
                     np.array(mean_training - std_training)[0],
                     np.array(mean_training + std_training)[0], alpha=.1, color='lightblue')

    # Validation
    plt.plot(mean_validation.tolist()[0], color='red', label='mean on validation')

    plt.fill_between(range(std_training.shape[1]),
                     np.array(mean_validation - std_validation)[0],
                     np.array(mean_validation + std_validation)[0], alpha=.1, color='pink')
