# data_preparer.py

import pandas as pd
import numpy as np

class DataPreparer_movielen:
    def __init__(self, file_path, sep='\t', user_col='user_id', item_col='item_id', rating_col='rating', timestamp_col='timestamp'):
        self.file_path = file_path
        self.sep = sep
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.timestamp_col = timestamp_col
        self.data = None
        self.user_map = None
        self.item_map = None

    def load_data(self):
        columns = [self.user_col, self.item_col, self.rating_col, self.timestamp_col]
        self.data = pd.read_csv(self.file_path, sep=self.sep, names=columns)
        self._reindex_ids()
        return self.data

    def _reindex_ids(self):
        # Create mappings from original IDs to new indices
        self.user_map = {original_id: new_id for new_id, original_id in enumerate(self.data[self.user_col].unique())}
        self.item_map = {original_id: new_id for new_id, original_id in enumerate(self.data[self.item_col].unique())}

        # Apply mappings to the data
        self.data[self.user_col] = self.data[self.user_col].map(self.user_map)
        self.data[self.item_col] = self.data[self.item_col].map(self.item_map)
