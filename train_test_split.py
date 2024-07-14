
# train_test_split.py

import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, ndcg_score
from funk_svd import FunkSVD

class TrainTestSplit:
    def __init__(self, model, test_size):
        self.model = model
        self.test_size = test_size
                
    def train_test(self, data):
       
        def detect_index_gaps(sequence):
             sequence = np.asarray(sequence)            
             # Ensure the sequence is sorted
             sorted_sequence = np.sort(sequence)            
             # Find the range of the sequence
             start, end = sorted_sequence[0], sorted_sequence[-1]            
             # Create a full range array
             full_range = np.arange(start, end + 1)            
             # Identify the gaps by finding elements in the full range not present in the sequence
             gaps = np.setdiff1d(full_range, sorted_sequence)
             print(len(gaps))
             return gaps
        
        #Partitioning
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
        

        # THIS IS THE PROBLEM AREA###########################


        # Count ratings for each user in the training set
        unique, counts = np.unique(train_ratings[:, 0], return_counts=True)
        user_rating_counts = dict(zip(unique, counts))
        
        # Identify users with less than 4 ratings
        users_to_remove = {user for user, count in user_rating_counts.items() if count < 4}

        # Remove these users from the training set
        train_mask = np.isin(train_ratings[:, 0], list(users_to_remove), invert=True)
        train_ratings = train_ratings[train_mask]

        # Remove these users from the test set
        test_mask = np.isin(test_ratings[:, 0], list(users_to_remove), invert=True)
        test_ratings = test_ratings[test_mask]

        # Filtering the test set - remove users or items from the test set that are not in the training set
        train_user_ids = set(train_ratings[:, 0].astype(int))
        train_item_ids = set(train_ratings[:, 1].astype(int))

        test_user_ids = test_ratings[:, 0].astype(int)
        test_item_ids = test_ratings[:, 1].astype(int)

        user_mask = np.isin(test_user_ids, list(train_user_ids))
        item_mask = np.isin(test_item_ids, list(train_item_ids))
        mask = user_mask & item_mask

        test_ratings = test_ratings[mask]

        # Reindexing user and item IDs to remove gaps
        unique_train_users = np.unique(train_ratings[:, 0].astype(int))
        unique_train_items = np.unique(train_ratings[:, 1].astype(int))

        user_id_map = {old: new for new, old in enumerate(unique_train_users)}
        item_id_map = {old: new for new, old in enumerate(unique_train_items)}

        train_ratings[:, 0] = np.vectorize(user_id_map.get)(train_ratings[:, 0].astype(int))
        train_ratings[:, 1] = np.vectorize(item_id_map.get)(train_ratings[:, 1].astype(int))
        test_ratings[:, 0] = np.vectorize(user_id_map.get)(test_ratings[:, 0].astype(int))
        test_ratings[:, 1] = np.vectorize(item_id_map.get)(test_ratings[:, 1].astype(int))

        def detect_index_gaps(sequence):
            sequence = np.asarray(sequence)
            sorted_sequence = np.sort(sequence)
            start, end = sorted_sequence[0], sorted_sequence[-1]
            full_range = np.arange(start, end + 1)
            gaps = np.setdiff1d(full_range, sorted_sequence)
            return gaps

        # Check for gaps in IDs
        train_user_gaps = detect_index_gaps(train_ratings[:, 0].astype(int))
        train_item_gaps = detect_index_gaps(train_ratings[:, 1].astype(int))
        test_user_gaps = detect_index_gaps(test_ratings[:, 0].astype(int))
        test_item_gaps = detect_index_gaps(test_ratings[:, 1].astype(int))

        print("Gaps in train user IDs:", train_user_gaps)
        print("Gaps in train item IDs:", train_item_gaps)
        print("Gaps in test user IDs:", test_user_gaps)
        print("Gaps in test item IDs:", test_item_gaps)

        ##########################################################
        

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
        test_true = np.array(test_ratings[:, 2])

        # Calculating RMSE and MAE metrics
        rmse = np.sqrt(mean_squared_error(test_true, test_preds))
        mae = mean_absolute_error(test_true, test_preds)
        
        # Converting the preds and true ratings to numpy arrays suitable for the ndcg
        true_ratings = [np.array(test_true)]
        pred_ratings = [np.array(test_preds)]
        ndcg = ndcg_score(true_ratings, pred_ratings)
        
        # Printing and logging the final loss metric values
        final_error_message = f"Train / Test RMSE: {rmse:.4f}, MAE: {mae:.4f}, NDCG: {ndcg:.4f}"
        print(final_error_message)
        logging.info(final_error_message)