
# main.py

import os
import logging
from datetime import datetime
from funk_svd import FunkSVD
from data_preparer_lfm import DataPreparer_lastfm
from data_preparer_mlen import DataPreparer_movielen
from train_test_split import TrainTestSplit
from timeit import default_timer as timer

# Logging config
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join('results', f'training_{timestamp}.log')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_file, filemode='w')

# Parameters
n_factors = 17
learning_rate = 0.002
regularization = 0.08
n_epochs = 15
test_size = 0.20

# Log the params
logging.info("Parameters:")
logging.info(f"n_factors: {n_factors}")
logging.info(f"learning_rate: {learning_rate}")
logging.info(f"regularization: {regularization}")
logging.info(f"n_epochs: {n_epochs}")
logging.info(f"test_size: {test_size}")
logging.info("/n")

# Used for preparing the LastFM data
# file_path = os.path.join(os.path.dirname(__file__), 'datasets', 'lastfm_reindex.csv')
# data_preparer = DataPreparer_lastfm(file_path)
# data = data_preparer.load_data()
# print(data.head())

# Used for preparing the MovieLen data 
file_path = os.path.join(os.path.dirname(__file__), 'datasets', 'movielen_reindex.data')
data_preparer = DataPreparer_movielen(file_path)
data = data_preparer.load_data()
print(data.head())

# Use this line to save any processed data for easier future use
# data.to_csv('PATH', sep = '\t', index=False )

# Initialize FunkSVD model
svd = FunkSVD(n_factors=n_factors, learning_rate=learning_rate, regularization=regularization, n_epochs=n_epochs)

# Train/Test Split method
tt_start_time = timer()
train_test_split = TrainTestSplit(model=svd, test_size=test_size)
train_test_split.train_test(data)

# Printing and logging some useful info
print("Train/Test Split time: ", timer() - tt_start_time)
print("Total time: ", timer() - tt_start_time)
logging.info(f"Train/Test Split time: {timer() - tt_start_time}")
logging.info(f"Total time: {timer() - tt_start_time}")
