import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import logging
from sklearn.metrics import ndcg_score
import pandas as pd

class FunkSVD:
    def __init__(self, n_factors, learning_rate, regularization, n_epochs):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.rmse_values = []  
       
    def fit(self, ratings):
        # Grab some basic stats
        self.mean_rating = np.mean(ratings[:, 2])
        self.std_rating = np.std(ratings[:, 2])
        

        # PROBLEM AREA
        
        
        # Number of users and items
        self.num_users = len(np.unique(ratings[:, 0])) + 1
        self.num_items = len(np.unique(ratings[:, 1])) + 1
             
        # Initialize with a smaller scale under a gaussian distribution
        # This is straight from the surprise package
        scale = 1 / self.n_factors 
        self.user_matrix = np.random.normal(loc=0, scale=scale, size=(self.num_users, self.n_factors))
        self.item_matrix = np.random.normal(loc=0, scale=scale, size=(self.num_items, self.n_factors)) 
        
        self.user_matrix += np.random.normal(loc=0, scale=1e-4, size=(self.num_users, self.n_factors)) 
        self.item_matrix += np.random.normal(loc=0, scale=1e-4, size=(self.num_items, self.n_factors)) 
        
        # Getting rid of any excessively large or small values
        np.clip(self.user_matrix, -1, 1, out=self.user_matrix)
        np.clip(self.item_matrix, -1, 1, out=self.item_matrix)
     
        # Checking for NANs at initialization
        assert not np.any(np.isnan(self.user_matrix)), "user_matrix contains NaNs after initialization"
        assert not np.any(np.isnan(self.item_matrix)), "item_matrix contains NaNs after initialization"
        
        # Convert the matrices to float 32 for parallel processing on cpu
        ratings[:, 0] = ratings[:, 0].astype(np.float32)
        ratings[:, 1] = ratings[:, 1].astype(np.float32)
        ratings[:, 2] = ratings[:, 2].astype(np.float32)
          
        # PROBLEM: Not hitting the train method, it just skips right over it.
        # Perform the training
        self.user_matrix, self.item_matrix, self.rmse_values = self.train(ratings, self.user_matrix, self.item_matrix, self.learning_rate, self.regularization, self.n_epochs)
         
        ##########################################

    # Huge thing i encountered here. numba/njit cannot access self, or anything instanced, must be static if used  
    @staticmethod
    @njit(parallel=True)
    def train(ratings, user_matrix, item_matrix, learning_rate, regularization, n_epochs):
        num_users, n_factors = user_matrix.shape
        num_items, _ = item_matrix.shape
        rmse_values = np.zeros(n_epochs)

        for epoch in range(n_epochs):
            # prange is used for paraelell indexing, that is very very cool.
            for i in range(ratings.shape[0]):
                user_id = (ratings[i, 0])
                item_id = (ratings[i, 1])
                rating = ratings[i, 2]

                # Making predictions taking dot product of the matrices
                prediction = np.dot(user_matrix[user_id], item_matrix[item_id])
                
                #if np.isnan(prediction):
                #     raise ValueError(f"Prediction is a NAN")
                
                # Update user and item latent vectors using error
                # Gradient descent being used here, remember to speak about in paper
                error = rating - prediction
                
                #if np.isnan(error):
                #     raise ValueError(f"Error is a NAN : {error}")
                
                user_matrix[user_id] += learning_rate * (error * item_matrix[item_id] - regularization * user_matrix[user_id])
                item_matrix[item_id] += learning_rate * (error * user_matrix[user_id] - regularization * item_matrix[item_id])
                                
                #if np.isnan(user_matrix[user_id]).any():
                #    raise ValueError(f"NAN detected in user matrix: {user_matrix[user_id]}")
                #
                #if np.isnan(item_matrix[item_id]).any():
                #    raise ValueError(f"NAN detected in item matrix: {item_matrix[item_id]}")

            # Calculate loss metric (RMSE) per epoch
            predictions = np.zeros(ratings.shape[0])
            for j in range(ratings.shape[0]):
                user_id = int(ratings[j, 0])
                item_id = int(ratings[j, 1])
                predictions[j] = np.dot(user_matrix[user_id], item_matrix[item_id])
            training_loss = np.sqrt(np.mean((ratings[:, 2] - predictions) ** 2))
            rmse_values[epoch] = training_loss
            
            
        return user_matrix, item_matrix, rmse_values

    def predict(self, user_id, item_id):
        pred = np.dot(self.user_matrix[user_id], self.item_matrix[item_id])
        return pred

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