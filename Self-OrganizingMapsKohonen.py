from math import ceil

import numpy as np
import random
from typing import Union, Callable
from random import shuffle
import matplotlib.pyplot as plt


class Kohonen_Maps():
    def __init__(self,
                 input_neurons: tuple = (40, 40),
                 epochs: int = 20,
                 learning_rate: Union[float, Callable] = 0.01,
                 random_state: int = 42
                 ) -> None:
        self.n_input_neurons = input_neurons
        self.weights = None
        self.epochs = epochs
        self.learning_rate = learning_rate if isinstance(learning_rate, float) else 0.01
        self.lr_shedule = lambda epoch, lr: lr * np.exp(-0.05 * epoch) if isinstance(learning_rate, float) else learning_rate
        self.random_state = random_state

    def _get_distance_value(self,
                            x_vec: np.array) -> np.array:
        # euclidean
        distances = np.sqrt(((x_vec - self.weights) ** 2).sum(axis=2))
        return distances

    def _get_weight_distances(self,
                              best_weight_coordinate: tuple,
                              coordinates_grid: np.array,
                              ) -> np.array:
        t = np.sqrt((coordinates_grid // self.n_input_neurons[0] - best_weight_coordinate[0]) ** 2 + (
                    coordinates_grid % self.n_input_neurons[0] - best_weight_coordinate[1]) ** 2)
        return t

    def _neighbour_function(self,
                            t: int,
                            weight_distances: np.array
                            ) -> float:
        sigmoid_func = lambda x: 1 / (1 + np.exp(-x))
        neighborhood_radius = sigmoid_func(t)
        neigh_func_values = np.exp(-weight_distances / (2 * neighborhood_radius))
        return neigh_func_values

    def fit(self, X: np.array) -> None:
        random.seed(self.random_state)
        np.random.seed(self.random_state)

        X_train = X.copy()
        random_sep = 1 / (self.n_input_neurons[0] * self.n_input_neurons[1])
        self.weights = np.random.uniform(low=-random_sep, high=random_sep,
                                         size=(self.n_input_neurons[0], self.n_input_neurons[1], X.shape[1]))
        coordinates_grid = np.arange(self.n_input_neurons[0] * self.n_input_neurons[1]).reshape(self.n_input_neurons[0],
                                                                                                self.n_input_neurons[1])

        lr = self.learning_rate

        for epoch in range(self.epochs):
            shuffle(X_train)
            scd = 0.0
            lr = self.lr_shedule(epoch, lr)

            for x_vec in X_train:
                # x_vec_num = np.random.randint(low=0, high=X.shape[0])
                # x_vec = X[x_vec_num]

                distances_x_weights = self._get_distance_value(x_vec)

                best_neuron = np.argmin(distances_x_weights)
                best_neuron_num = (best_neuron // X_train.shape[0], best_neuron % X_train.shape[0])
                best_neuron_distances = self._get_weight_distances(best_neuron_num, coordinates_grid)

                neigh_func_values = self._neighbour_function(epoch, best_neuron_distances)

                cluster_diff = neigh_func_values[:, :, np.newaxis] * (x_vec - self.weights)
                clusterization_unstable = np.linalg.norm(cluster_diff)

                scd += clusterization_unstable

                self.weights += lr * cluster_diff
                # self.w_inpout += np.outer(neigh_func_values, (x_vec - self.w_inpout)).mean(axis=0)
            print(epoch, ": ", scd / X_train.shape[0])

    def show(self):
        fig, ax = plt.subplots(nrows=ceil(self.weights.shape[-1] / 4), ncols=4, figsize=(24, 8),
                               subplot_kw=dict(xticks=[], yticks=[]))
        for i in range(ceil(self.weights.shape[-1] / 4)):
            for j in range(4):
                if (i + 1) * (j + 1) <= self.weights.shape[-1]:
                    ax[i][j].imshow(self.weights[:, :, 4 * i + j])