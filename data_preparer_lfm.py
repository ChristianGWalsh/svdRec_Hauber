
# data_preparer_lastfm.py

import pandas as pd
import numpy as np


class DataPreparer_lastfm:
    def __init__(self, file_path, sep='\t', user_col='user_id', item_col='artist_id', rating_col='plays', artist_name_col='artist_name'):
        self.file_path = file_path
        self.sep = sep
        self.user_col = user_col
        self.item_col = item_col
        self.artist_name_col = artist_name_col
        self.rating_col = rating_col
        self.data = None
        self.user_map = None
        self.item_map = None
    
   
    def load_data(self):
        # Define the column names based on the new format
        columns = [self.user_col, self.item_col, self.artist_name_col, self.rating_col]
        
        # Specify to not use the artist name column, they are strings
        self.data = pd.read_csv(self.file_path, sep=self.sep, usecols=[self.user_col, self.item_col, self.rating_col])
        
        # This ensures that the indices for both users and items are zero based and continuous
        self._reindex_ids()
        return self.data

    def _reindex_ids(self):
        # Create mappings from original IDs to new indices
        self.user_map = {original_id: new_id for new_id, original_id in enumerate(self.data[self.user_col].unique())}
        self.item_map = {original_id: new_id for new_id, original_id in enumerate(self.data[self.item_col].unique())}

        # Apply mappings to the data
        self.data[self.user_col] = self.data[self.user_col].map(self.user_map)
        self.data[self.item_col] = self.data[self.item_col].map(self.item_map)


