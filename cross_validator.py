# cross_validator.py

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from funk_svd import FunkSVD
import logging

class CrossValidator:
    def __init__(self, model, n_splits=5):
        self.model = model
        self.n_splits = n_splits

    def cross_validate(self, ratings):
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        rmse_scores = []
        mae_scores = []

        for fold, (train_index, test_index) in enumerate(kf.split(ratings), 1):
            print(f"Fold {fold} of {kf.get_n_splits()}")

            train_ratings, test_ratings = ratings[train_index], ratings[test_index]

            #fit the model here (training)
            # this function alone is using the gpu
            self.model.fit(train_ratings)

            #make predictions
            test_preds = []
            for user_id, item_id, _ in test_ratings:
                if user_id < self.model.user_matrix.shape[0] and item_id < self.model.item_matrix.shape[0]:
                    test_preds.append(self.model.predict(user_id, item_id))
                else:
                    test_preds.append(np.mean(train_ratings[:, 2]))  # use the mean rating if index is out of bounds

            test_preds = np.array(test_preds)
            test_true = test_ratings[:, 2]

            #calculating error RMSE and MAE
            rmse = np.sqrt(mean_squared_error(test_true, test_preds))
            mae = mean_absolute_error(test_true, test_preds)

            rmse_scores.append(rmse)
            mae_scores.append(mae)

        final_rmse_message = f"Cross Validation RMSE: {np.mean(rmse_scores):.4f}"
        final_mae_message = f"Cross Validation MAE: {np.mean(mae_scores):.4f}"
        print(final_rmse_message)
        print(final_mae_message)
        logging.info(final_rmse_message)
        logging.info(final_mae_message)