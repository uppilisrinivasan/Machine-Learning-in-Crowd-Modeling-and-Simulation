
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
    
    col_names = ["timestep", "ID", "X", "Y", "Z"]
    data = pd.read_csv(path, sep=" ", header=None, names=col_names)
    
    data.drop(labels='Z', inplace=True, axis=1)
    
    return data

#Finding the k-nnearest neighboring points and distances for points in the same time step
def finding_knn(positions,num_neighbors):
    """
    Reads the current position of all points at a particular timestep
    Returns the distances and indices of the k-nearest neighbors
    """
    nbrs = NearestNeighbors(n_neighbors= num_neighbors+1, algorithm='ball_tree').fit(positions)
    distances, indices = nbrs.kneighbors(positions)
    indices = indices[:,1:]
    distances = distances[:,1:]
    
    return distances, indices

#Calculating mean spacing
def calculating_mean_spacing(positions, num_neighbors):
    """
    Reads the current position of all points at a particular timestep
    Returns mean spacing for each point
    """
    distances, indices = finding_knn(positions,num_neighbors)

    mean_spacing = np.average(distances, axis=1, keepdims=True)

    return mean_spacing

#X,Y Co-ordinates of K-nearest neighbors
def knn_positions(positions,num_neighbors):
    distances, indices = finding_knn(positions,num_neighbors)
    rows = len(indices)
    cols = len(indices[0])
    rel_pos = np.zeros(shape=(rows, cols, 2))
    for i in range(rows):
        for j in range(cols):
            temp = indices[i][j]
            rel_pos[i][j] = positions[temp]
            #rel_pos[i][j][1] = pos[temp][1]

    return rel_pos

original_data_path = "D:/TUM/SEM3/Praktikum/Project/crowd-modeling-main/final_project/data/raw/test.txt"
num_neighbors=5

col_names = ["timestep", "ID", "X", "Y", "Z"]

data = read_data(original_data_path)
data=data.iloc[1:]

#Converting daataframe to numpy arrays separating timestep, id and position
ts = data[['timestep']].to_numpy()
ped_id = data[['ID']].to_numpy()
pos = data[['X','Y']].to_numpy()
a=len(ts)
speed = np.empty(a)
prev_speed = np.empty(500)

# Calculating speed for the pedestrian
# If the pedestrian reaches the target in the current timestep (ie there is no pedestrian with the same id in next timestep),
# the speed is considered as the speed in the previous timestep 
for i in range(len(ts)):
    curr_ts = int(ts[i])
    curr_ped_id = int(ped_id[i])
    curr_posit_x = float(pos[i][0])
    curr_posit_y = float(pos[i][1])
    
    for j in range(i,len(ts)):
        if int(ts[j]) == curr_ts + 1 and int(ped_id[j]) == curr_ped_id:
            new_posit_x = float(pos[j][0])
            new_posit_y = float(pos[j][1])
            
            p= np.array([curr_posit_x,curr_posit_y])
            q= np.array([new_posit_x, new_posit_y])
            d=np.linalg.norm(p-q)
            
            speed[i] = d/0.4
            prev_speed[curr_ped_id] = speed[i]
            break

        else:
            speed[i] = prev_speed[curr_ped_id]




mean_spacing = np.empty(a)
dist = np.empty(shape=[a,num_neighbors])
index = np.empty(shape=[a,num_neighbors])
rel_positions = np.empty(shape=[a,num_neighbors,2])
final_ts = int(ts[-1])
count = 0

#Since vadere output has time step as integer coordinates we can use the last time step for range
#We separate the position data wrt timestep and process 1 timestep at a time
for i in range(1, final_ts+1):
    peds = []
    curr_pos = []
    dis =[]
    indi=[]
    ms=[]
    knn_rp=[]
    for j in range(len(ts)):
        if int(ts[j]) == i:
            peds.append(ped_id[j])
            curr_pos.append(pos[j])
    
    peds = np.array(peds)
    curr_pos = np.array(curr_pos)
    
    dis, indi = finding_knn(curr_pos,num_neighbors)
    ms = calculating_mean_spacing(curr_pos,num_neighbors)
    knn_rp = knn_positions(curr_pos,num_neighbors)
   
    #adding the mean spacing and knn positions from chunks to main arry
    for k in range(len(dis)):
        dist[count+k] = dis[k]
        index[count+k] = indi[k]
        mean_spacing[count+k] = ms[k]
        rel_positions[count+k] = knn_rp[k]
        if k == len(dis)-1:
            count=count+k+1

time =  np.reshape(ts, (np.product(ts.shape),))
ids = np.reshape(ped_id, (np.product(ped_id.shape),))

spacing = np.reshape(mean_spacing, (np.product(mean_spacing.shape),))

rel = rel_positions

#Calculating relative positions of KNN points wrt to current pedestriana position
for i in range(len(rel)):
    for j in range(num_neighbors):
        rel[i][j][0]=rel[i][j][0]-float(pos[i][0])
        rel[i][j][1]=rel[i][j][1]-float(pos[i][1])

mat=np.reshape(rel, (rel.shape[0],rel.shape[1]*rel.shape[2]))

#Combining the numpy arrays to dataframe to create the datset
dataset = pd.DataFrame({'timestep' : time, 'ID' : ids })
dataset = pd.concat([dataset,pd.DataFrame(pos)], axis=1)
dataset.columns = ['timestep','ID', 'X','Y']

dataset['speed'] = speed.tolist()
dataset['mean_spacing'] = spacing.tolist()
dataset = pd.concat([dataset, pd.DataFrame(mat)], axis=1)

dataset.to_csv(r'D:/TUM/SEM3/Praktikum/Project/crowd-modeling-main/final_project/data/raw/test_dataset.txt', header=True, index=None, sep=' ', mode='a')
#print(dataset)



