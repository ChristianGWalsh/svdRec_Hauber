
# train_test_split.py

import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from funk_svd import FunkSVD

class TrainTestSplit:
    def __init__(self, model, test_size):
        self.model = model
        self.test_size = test_size

    def train_test(self, data):
        train_data, test_data = train_test_split(data, test_size=self.test_size, random_state=42)
        
        # This has is a very simple fix, will come later
        # Use for movieLen
        # train_ratings = train_data[['user_id', 'item_id', 'rating']].values
        # test_ratings = test_data[['user_id', 'item_id', 'rating']].values
        
        # Use for LastFM
        # train_ratings = train_data[['user_id', 'artist_id', 'plays']].values
        # test_ratings = test_data[['user_id', 'artist_id', 'plays']].values
        
        # Use for MillionSong
        train_ratings = train_data[['user_id', 'song_id', 'plays']].values
        test_ratings = test_data[['user_id', 'song_id', 'plays']].values

        # Fit the model to the training data
        self.model.fit(train_ratings)
   
        # Predict the ratings for the test data
        test_preds = []
        for user_id, item_id, _ in test_ratings:
            if user_id < self.model.user_matrix.shape[0] and item_id < self.model.item_matrix.shape[0]:
                test_preds.append(self.model.predict(user_id, item_id))
            else:
                test_preds.append(np.mean(train_ratings[:, 2]))  # use the mean rating if index is out of bounds

        test_preds = np.array(test_preds)
        test_true = test_ratings[:, 2]

        # Calculating RMSE and MAE metrics
        rmse = np.sqrt(mean_squared_error(test_true, test_preds))
        mae = mean_absolute_error(test_true, test_preds)
        final_rmse_message = f"Train / Test RMSE: {rmse:.4f}, MAE: {mae:.4f}"
        print(final_rmse_message)
        logging.info(final_rmse_message)