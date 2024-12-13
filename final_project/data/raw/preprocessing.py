import pandas as pd
import numpy as np
from scipy.spatial import distance

def remove_diagonal(A):
    return A[~np.eye(A.shape[0],dtype=bool)].reshape(A.shape[0],-1)

def create_dataset_corridor(path_to_data, num_neighbors = 5):
  NO_NEIGHBORS = num_neighbors
  TIMESTEP_LENGTH = 0.4

  dic = {
      'timestep': [],
      'pid': [],
      'pos': [],
      'mean_spacing': [],
      'speed': [],
      'knn': []
  }

  columns = ["ID", "FRAME", "X", "Y", "Z"]
  data = pd.read_csv(path_to_data, sep=" ", header=None)

  data.columns = columns
  dataset= data.drop(['Z'], axis=1)

  for timestep in range(0, dataset['FRAME'].max()):
    frame = dataset[dataset['FRAME'] == timestep]
    next_frame = dataset[dataset['FRAME'] == timestep + 1]

    frame = frame[frame['ID'].isin(next_frame['ID'])]
    next_frame = next_frame[next_frame['ID'].isin(frame['ID'])]

    if len(frame) <= NO_NEIGHBORS:
        continue

    pos = frame[['X', 'Y']].to_numpy()
    next_pos = next_frame[['X', 'Y']].to_numpy()

    dist = distance.squareform(distance.pdist(pos))
    dist = remove_diagonal(dist)[:, :NO_NEIGHBORS]

    knn = np.argsort(dist, axis=1)

    dic['timestep'] += frame['FRAME'].to_list()
    dic['pid'] += frame['ID'].to_list()
    dic['pos'] += pos.tolist()
    dic['mean_spacing'] += dist.mean(axis = 1).tolist()
    dic['speed'] += (np.linalg.norm(next_pos - pos, axis=1) / TIMESTEP_LENGTH).tolist() 
    dic['knn'] += (pos[knn] - pos[:, np.newaxis]).tolist()


    data = pd.DataFrame(dic)
    return data

def convert_df(df):
    new_df = df[["timestep", "pid", "speed", "mean_spacing"]].copy()
    new_df["ID"] = df["pid"]
    new_df["X"] = df["pos"].apply(lambda x: x[0])
    new_df["Y"] = df["pos"].apply(lambda x: x[1])
    
    for i in range(5):
        new_df[str(2 * i)] = df["knn"].apply(lambda x: x[i][0])
        new_df[str(2 * i + 1)] = df["knn"].apply(lambda x: x[i][1])
    
    return new_df
