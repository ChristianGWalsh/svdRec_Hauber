import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange, jit
import logging

class FunkSVD:
    def __init__(self, n_factors, learning_rate, regularization, n_epochs):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.rmse_values = []

    def fit(self, ratings):
        
       # Converting plays to rating
       # This is subject to change
        for plays in ratings[:,2]:
           if plays >= 1 and plays <= 5:
               plays = 1
           elif plays >= 6 and plays <= 10:
               plays = 2
           elif plays >= 11 and plays <= 20:
               plays = 3
           elif plays >= 21 and plays <= 50:
               plays = 4
           elif plays >= 51:
               plays = 5
                
        self.mean_rating = np.mean(ratings[:, 2])
        self.std_rating = np.std(ratings[:, 2])
        
        # Normalize ratings
        # Changed but put back.  The unused was suggested, but metrics seem to be better when using this.
        #ratings[:, 2] = (ratings[:, 2] - .5) / (5 - .5)
        ratings[:, 2] = (ratings[:, 2] - self.mean_rating) / self.std_rating

        # Number of users and items
        self.num_users = int(np.max(ratings[:, 0])) + 1
        self.num_items = int(np.max(ratings[:, 1])) + 1
        
        # Changed
        # Size for the shape of the matrix to be filled, in this case, number of users by number of features
        # Scale is the spread for the distrobution (gaussian). 1 is default
        self.user_matrix = np.random.normal(scale=.1,size=(self.num_users, self.n_factors))
        self.item_matrix = np.random.normal(scale=.1, size=(self.num_items, self.n_factors))

        # Convert the matrices to float 32 for parallel processing on cpu
        ratings[:, 0] = ratings[:, 0].astype(np.float32)
        ratings[:, 1] = ratings[:, 1].astype(np.float32)
        ratings[:, 2] = ratings[:, 2].astype(np.float32)

        # Perform the training
        self.user_matrix, self.item_matrix, self.rmse_values = self.train(ratings, self.user_matrix, self.item_matrix, self.learning_rate, self.regularization, self.n_epochs)

    # Huge thing i encountered here.  numba cannot self, or anything instanced. 
    # Making the method static allows for the method to compile properly, and access the class    
    @staticmethod
    @njit(parallel=True)
    def train(ratings, user_matrix, item_matrix, learning_rate, regularization, n_epochs):
        num_users, n_factors = user_matrix.shape
        num_items, _ = item_matrix.shape
        rmse_values = np.zeros(n_epochs)

        for epoch in range(n_epochs):
            # prange is used for paraelell indexing, that is very very cool.
            for i in prange(ratings.shape[0]):
                user_id = int(ratings[i, 0])
                item_id = int(ratings[i, 1])
                rating = ratings[i, 2]

                # Making predictions taking dot product of the matrices
                prediction = np.dot(user_matrix[user_id], item_matrix[item_id])

                # Update user and item latent vectors using error
                # Gradient descent being used here, remember to speak about in paper
                error = rating - prediction
                user_matrix[user_id] += learning_rate * (error * item_matrix[item_id] - regularization * user_matrix[user_id])
                item_matrix[item_id] += learning_rate * (error * user_matrix[user_id] - regularization * item_matrix[item_id])

            # Calculate loss metric (RMSE) per epoch
            predictions = np.zeros(ratings.shape[0])
            for j in prange(ratings.shape[0]):
                user_id = int(ratings[j, 0])
                item_id = int(ratings[j, 1])
                predictions[j] = np.dot(user_matrix[user_id], item_matrix[item_id])
            training_loss = np.sqrt(np.mean((ratings[:, 2] - predictions) ** 2))
            rmse_values[epoch] = training_loss

        return user_matrix, item_matrix, rmse_values

    # Why is this one different from the method inside the training loop?
    # This cannot go in the loop with the parallel processing, however leaving this version here lowers the metrics
    def predict(self, user_id, item_id):
        pred = np.dot(self.user_matrix[user_id], self.item_matrix[item_id])
        return pred * self.std_rating + self.mean_rating

    def get_user_matrix(self):
        return self.user_matrix

    def get_item_matrix(self):
        return self.item_matrix

    # This is busted needs fixing
    def plot_training_loss(self):
        plt.plot(range(1, self.n_epochs + 1), self.rmse_values, label='Training RMSE')
        plt.xlabel('Epochs')
        plt.ylabel('RMSE')
        plt.title('Training RMSE over Epochs')
        plt.legend()
        plt.savefig('training_rmse_plot.png')
        plt.show()