from typing import Union
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors

def read_data(path: Union[pd.DataFrame, str]) -> pd.DataFrame:
    """
    Reads the data from text file and returns the data as a dataframe
    """
    if isinstance(path, pd.DataFrame):
        return path
    
    col_names = ["ID", "timestep", "X", "Y", "Z"]
    data = pd.read_csv(path, sep=" ", header=None, names=col_names)
    
    data.drop(labels='Z', inplace=True, axis=1)

    data['X'] = data['X'] / 100
    data['Y'] = data['Y'] / 100
    
    return data

def create_dataset(original_data_path):
  data = read_data(original_data_path)
  columns_titles = ["timestep","ID", "X", "Y"]
  data=data.reindex(columns=columns_titles)
  sort = data.sort_values(data.columns[0], ascending = True)
  print(sort)
  return sort

#col_names = ["timestep", "ID", "X", "Y", "Z"]


#sort.to_csv(r'D:/TUM/SEM3/Praktikum/Project/crowd-modeling-main/final_project/data/raw/Corridor_Data/ug-180-230_preprocessing.txt', header=True, index=None, sep=' ', mode='a')
