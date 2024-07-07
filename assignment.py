import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
'''
file_path = 'datasets/train_triplets.txt'
column_names = ['user_id', 'song_id', 'play_count']
df = pd.read_csv(file_path, sep='\t', names=column_names)
print(df.head())
user_id_mapping = {original_id: new_id for new_id, original_id in enumerate(df['user_id'].unique())}
song_id_mapping = {original_id: new_id for new_id, original_id in enumerate(df['song_id'].unique())}
df['user_id'] = df['user_id'].map(user_id_mapping)
df['song_id'] = df['song_id'].map(song_id_mapping)
print("\nData with 0-based integer IDs:")
print(df.head())
output_file_path = 'datasets/train_triplets_processed.csv'
df.to_csv(output_file_path, index=False)
'''
file_path = 'datasets/train_triplets_processed.csv'
df = pd.read_csv(file_path)
print(df.head())
print("Basic Statistics for Play Counts:")
print(df['play_count'].describe())
highest_play_count = df['play_count'].max()
lowest_play_count = df['play_count'].min()
mean_play_count = df['play_count'].mean()
median_play_count = df['play_count'].median()
mode_play_count = df['play_count'].mode()[0]
print("\nAdditional Statistics:")
print(f"Highest Play Count: {highest_play_count}")
print(f"Lowest Play Count: {lowest_play_count}")
print(f"Mean Play Count: {mean_play_count}")
print(f"Median Play Count: {median_play_count}")
print(f"Mode Play Count: {mode_play_count}")
song_play_counts = df.groupby('song_id')['play_count'].sum().sort_values(ascending=False)
sample_size = 10000
sampled_indices = np.random.choice(song_play_counts.index, size=sample_size, replace=False)
sampled_song_play_counts = song_play_counts.loc[sampled_indices].sort_values(ascending=False)
plt.figure(figsize=(10, 6))
plt.bar(range(len(sampled_song_play_counts)), np.log1p(sampled_song_play_counts.values))
plt.title('Logarithmic Total Play Counts per Sampled Songs')
plt.xlabel('Sampled Song Rank')
plt.ylabel('Logarithmic Total Play Count')
plt.show()