
# main.py

import os
import logging
from datetime import datetime
from funk_svd import FunkSVD
from data_preparer_lfm import DataPreparer_lastfm
from data_preparer_mlen import DataPreparer_movielen
from train_test_split import TrainTestSplit
from timeit import default_timer as timer
import pandas as pd

# Logging config
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join('results', f'training_{timestamp}.log')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_file, filemode='w')

# Parameters
n_factors = 17
learning_rate = 0.002 # When going anything aboe this .002, the test predictions will be NAN's when using the lastfm data
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
#file_path = os.path.join(os.path.dirname(__file__), 'datasets', 'movielen_reindex.data')
#data_preparer = DataPreparer_movielen(file_path)
#data = data_preparer.load_data()

# Used for preparing the MillionSong data
#file_path = 'datasets/train_triplets.txt'
#data_preparer = DataPreparer_movielen(file_path)
#data = data_preparer.load_data()

# Use this line to save any processed data for easier future use
#data.to_csv('datasets/millionsong_reindex.csv', index=False )

# Grep processed data
file_path = 'datasets/millionsong_reindex.csv'
data = pd.read_csv(file_path)
print(data.head())
print(data.tail())

# Seeing some stats, using this to help with my conversion from plays to ratings
#highest_play_count = data['plays'].max()
#lowest_play_count = data['plays'].min()
#mean_play_count = data['plays'].mean()
#median_play_count = data['plays'].median()
#mode_play_count = data['plays'].mode()[0]
#print("\nAdditional Statistics:")
#print(f"Highest Play Count: {highest_play_count}")
#print(f"Lowest Play Count: {lowest_play_count}")
#print(f"Mean Play Count: {mean_play_count}")
#print(f"Median Play Count: {median_play_count}")
#print(f"Mode Play Count: {mode_play_count}")

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
