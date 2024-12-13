import math
from math import dist

import numpy as np
import scipy.spatial.distance
from PIL import Image, ImageTk



def euc_distance(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


class Pedestrian:
    """
    Defines a single pedestrian.
    """

    def __init__(self, position, dist_per_timestep, scenario):
        self._position = position
        self.dist_per_timestep = dist_per_timestep
        self.unused_distance = 0
        self.scenario = scenario

        scenario.grid[position[0], position[1]] = Scenario.NAME2ID['PEDESTRIAN']
 
    @property
    def position(self):
        return self._position

    def get_neighbors(self):
        """
        Compute all neighbors in a 9 cell neighborhood of the current position.
        :param scenario: The scenario instance.
        :return: A list of neighbor cell indices (x,y) around the current position.
        """
        return [
            (int(x + self._position[0]), int(y + self._position[1]))
            for x in [-1, 0, 1]
            for y in [-1, 0, 1]
            if 0 <= x + self._position[0] < self.scenario.width and 0 <= y + self._position[1] < self.scenario.height and np.abs(x) + np.abs(y) > 0
        ]

    def substep(self):
        """
        Moves to the cell with the lowest distance to the target.

        :param scenario: The current scenario instance.
        :returns: Whether the pedestrian moved during this substep.
        """
        neighbors = self.get_neighbors()
        next_cell_distance = self.scenario.target_distance_grids[self._position[0]][self._position[1]]
        next_pos = self._position
        stepped = False

        for (n_x, n_y) in neighbors:
            if next_cell_distance > self.scenario.target_distance_grids[n_x, n_y] and \
               self.scenario.grid[n_x, n_y] == Scenario.NAME2ID['EMPTY'] and \
               self.unused_distance >= euc_distance(self._position, (n_x, n_y)):
                
                stepped = True
                next_pos = (n_x, n_y)
                next_cell_distance = self.scenario.target_distance_grids[n_x, n_y]
        
        if stepped:
            self.unused_distance -= euc_distance(self._position, next_pos)
            self.scenario.grid[self._position[0], self._position[1]] = Scenario.NAME2ID['EMPTY']
            self._position = next_pos
            self.scenario.grid[self._position[0], self._position[1]] = Scenario.NAME2ID['PEDESTRIAN']

        return stepped

    def start_step(self):
        """
        Initializes a new step by allowing the pedestrian to move @self.dist_per_timestep distance.
        """
        self.unused_distance = self.dist_per_timestep


class Scenario:
    """
    A scenario for a cellular automaton.
    """
    GRID_SIZE = (500, 500)
    ID2NAME = {
        0: 'EMPTY',
        1: 'TARGET',
        2: 'OBSTACLE',
        3: 'PEDESTRIAN'
    }
    NAME2COLOR = {
        'EMPTY': (255, 255, 255),
        'PEDESTRIAN': (255, 0, 0),
        'TARGET': (0, 0, 255),
        'OBSTACLE': (255, 0, 255)
    }
    NAME2ID = {
        ID2NAME[0]: 0,
        ID2NAME[1]: 1,
        ID2NAME[2]: 2,
        ID2NAME[3]: 3
    }

    def __init__(self, width, height, targets, obstacles, pedestrians, use_dijkstra):
        """"
        Initializes a simulation given all the required components and computes the target distance grids.
        """

        if width < 1 or width > 1024:
            raise ValueError(f"Width {width} must be in [1, 1024].")

        if height < 1 or height > 1024:
            raise ValueError(f"Height {height} must be in [1, 1024].")

        self.width = width
        self.height = height
        self.grid_image = None
        self.grid = np.zeros((width, height))

        for target in targets:
            self.grid[target[0], target[1]] = Scenario.NAME2ID['TARGET']

        for obstacle in obstacles:
            self.grid[obstacle[0], obstacle[1]] = Scenario.NAME2ID['OBSTACLE']

        self.pedestrians = [Pedestrian(ped['position'], ped['desired_distance'], self) for ped in pedestrians]

        self.use_dijkstra = use_dijkstra

        self.target_distance_grids = self.recompute_target_distances()

    def recompute_target_distances(self):
        self.target_distance_grids = self.update_target_grid()
        return self.target_distance_grids
    
    #finds index of a point(pos2) from the grid or positions(pos1)
    def get_index_matching(self,pos1,pos2):
            for num in range(0,pos1.shape[0]):
                if pos1[num,0] == pos2[0] and pos1[num,1] == pos2[1]:
                    return num
                else:
                    continue
    #gets neighbors for a node 
    def get_node_neighbors(self,node):
        """
        Compute all neighbors in a 9 cell neighborhood of the current position.
        :param : node.
        :return: A list of neighbor cell indices (x,y) around the current position.
        """
        return [
            (int(x + node[0]), int(y + node[1]))
            for x in [-1, 0, 1]
            for y in [-1, 0, 1]
            if 0 <= x + node[0] < self.width and 0 <= y + node[1] < self.height and np.abs(x) + np.abs(y) > 0
        ]
    
    #Calculate distance matrix for the grid using euclidean distance. Used for chicken test scenario 
    def euclidean_distance(self):
        """
        Computes distance matrix using euclidean distance
        :return : distance matrix for all positions wrt to all targets taking into account the obstacles
        """
        targets = self.targets
        obstacles = self.obstacles
        positions = self.positions
      
        tar = np.row_stack(targets)
        if len(obstacles)!=0:
            obs = np.row_stack(obstacles)
        else:
            obs = []
        
        # compute the pair-wise distances in one step with scipy.
        distance = scipy.spatial.distance.cdist(tar, positions)
        
        tar_row = tar.shape[0]
        
        #Penalizing the distance for the locations where obstacles are present
        for t in range(0,tar_row):
            if len(obs) != 0:
                pos_row = positions.shape[0]
                obs_row = obs.shape[0]

                for z in range(0,obs_row):
                    idx_obs = self.get_index_matching(positions,obs[z])
                    distance[t,idx_obs]+=100000
        
        return distance
    
    #Calculating distance matrix using dijkstra algorithm    
    def dijkstra(self):
        """
        Computes distance matrix using dijkstra algorithm
        :return: distance matrix for all points wrt to all targets taking into account the obstacles
        """
        targets = self.targets
        obstacles = self.obstacles
        positions = self.positions
        
        if len(obstacles)!=0:
            obs = np.row_stack(obstacles)
        else:
            obs = []
        tar = np.row_stack(targets)
        tar_row = tar.shape[0]
        
        if len(obs) !=0:
            obs_row = obs.shape[0]
        else:
            obs_row = 0

        s=(len(tar),len(positions))
        distance=np.zeros(s)
        
        #Set distance to a high value for all nodes
        for i in range(0,distance.shape[0]):
            for j in range(0,distance.shape[1]):
                distance[i,j]=100000
        
        #Setting unvisted state as 1 for all nodes initially
        unvisited_state = np.ones(positions.shape[0])
        
        idx_of_targets=[]
        idx_of_obstacles =[]

        #Initializing distance matrix for targets. Setting distance for targets as 0 and changing unvisited state (ie mark them as visited)
        for t in range(0,tar_row):
            a=self.get_index_matching(positions,tar[t])
            idx_of_targets.append(a)

        for idx in idx_of_targets:
            unvisited_state[idx]=0
            distance[0,idx]=0

        #Changing unvisited_state for obstacles
        for o in range(0,obs_row):
            b=self.get_index_matching(positions,obs[o])
            idx_of_obstacles.append(b)

        for id in idx_of_obstacles:
            unvisited_state[id]=0

        #Creating unvisited nodes (which have unvisited_state==1)
        unvisited_nodes=[]
        visited_nodes=[]

        for idx in range(0,len(unvisited_state)):
            a=positions[idx]
            if unvisited_state[idx]==1:
                unvisited_nodes.append(a)        
            else:
                visited_nodes.append(a)
        
        unvisited_nodes=np.array(unvisited_nodes)
        visited_nodes = np.array(visited_nodes)
        unvisited = True

        #Changing distances of unvisited nodes by taking into account the distances of neighbors
        for t in range(0,tar_row):
            while(unvisited):
                for node in unvisited_nodes:
                    idx_node = self.get_index_matching(positions,node)
                    index_node = self.get_index_matching(unvisited_nodes,node)
                
                #Checking distance of neighbors from current node
                    neighbors= self.get_node_neighbors(node)
                    current_dist = distance[t,idx_node]
                    new_dist=[]
                    neighbor_state=[]
                    result = False
                
                #Finding closest neighbor and calculating minimum distance
                    for neighbor in neighbors:
                        idx_neighbor=self.get_index_matching(positions,neighbor)
                        euc_dist = euc_distance(node,neighbor)
                        d=euc_dist + distance[t,idx_neighbor]
                        new_dist.append(d)
                        min_dist = min(new_dist)
                        e=unvisited_state[idx_neighbor]
                        neighbor_state.append(e)
                    if min_dist < current_dist:
                        distance[t,idx_node] = min_dist
                        unvisited_state[idx_node]=0
                
                #Check visited state of all neighbors so that we can remove the node from unvisited queue
                    result = all(elem == 0 for elem in neighbor_state)
                    if result:
                        unvisited_nodes=np.delete(unvisited_nodes,index_node,0)
                    
                if len(unvisited_nodes)==0:
                    unvisited = False
        return distance
    
    
    def update_target_grid(self):
        """
        Computes the shortest distance from every grid point to the nearest target cell.
        :returns: The distance for every grid cell, as a np.ndarray.
        """
        targets = []
        for x in range(self.width):
            for y in range(self.height):
                if self.grid[x, y] == Scenario.NAME2ID['TARGET']:
                    targets.append([x, y])  # y and x are flipped because they are in image space.
        if len(targets) == 0:
            return np.zeros((self.width, self.height))

        obstacles = []
        for x in range(self.width):
            for y in range(self.height):
                if self.grid[y, x] == Scenario.NAME2ID['OBSTACLE']:
                    obstacles.append([y, x])  # y and x are flipped because they are in image space.
        
        #targets = np.row_stack(targets)
        x_space = np.arange(0, self.width)
        y_space = np.arange(0, self.height)
        xx, yy = np.meshgrid(x_space, y_space)

        positions = np.column_stack([xx.ravel(order='F'), yy.ravel(order='F')])

        self.targets = targets
        self.obstacles = obstacles
        self.positions = positions
        
        # after the target positions and all grid cell positions are stored,
        #Compute distance matrix using either euclidean or dijkstra algorithm depending on state of use_dijkstra defined in the input file
        if self.use_dijkstra:
            distances = self.dijkstra()
        else:
            distances = self.euclidean_distance()
        
        # now, compute the minimum over all distances to all targets.
        distances = np.min(distances, axis=0)

        return distances.reshape((self.width, self.height))

    def start_step(self):
        for pedestrian in self.pedestrians:
            pedestrian.start_step()

    def substep(self):
        """
        Updates the position of all pedestrians.

        :returns: Whether any pedestrian moved during this substep.
        """
        stepped = False

        for pedestrian in self.pedestrians:
            stepped |= pedestrian.substep()

        return stepped

    @staticmethod
    def cell_to_color(_id):
        return Scenario.NAME2COLOR[Scenario.ID2NAME[_id]]

    def target_grid_to_image(self, canvas, old_image_id):
        """
        Creates a colored image based on the distance to the target stored in
        self.target_distance_gids.
        :param canvas: the canvas that holds the image.
        :param old_image_id: the id of the old grid image.
        """
        im = Image.new(mode="RGB", size=(self.width, self.height))
        pix = im.load()
        for x in range(self.width):
            for y in range(self.height):
                target_distance = self.target_distance_grids[x][y]
                pix[x, y] = (max(0, min(255, int(10 * target_distance) - 0 * 255)),
                             max(0, min(255, int(10 * target_distance) - 1 * 255)),
                             max(0, min(255, int(10 * target_distance) - 2 * 255)))
        im = im.resize(Scenario.GRID_SIZE, Image.NONE)
        self.grid_image = ImageTk.PhotoImage(im)
        canvas.itemconfigure(old_image_id, image=self.grid_image)

    def to_image(self, canvas, old_image_id):
        """
        Creates a colored image based on the ids stored in self.grid.
        Pedestrians are drawn afterwards, separately.
        :param canvas: the canvas that holds the image.
        :param old_image_id: the id of the old grid image.
        """
        im = Image.new(mode="RGB", size=(self.width, self.height))
        pix = im.load()
        for x in range(self.width):
            for y in range(self.height):
                pix[x, y] = self.cell_to_color(self.grid[x, y])
        for pedestrian in self.pedestrians:
            x, y = pedestrian.position
            pix[x, y] = Scenario.NAME2COLOR['PEDESTRIAN']
        im = im.resize(Scenario.GRID_SIZE, Image.NONE)
        self.grid_image = ImageTk.PhotoImage(im)
        canvas.itemconfigure(old_image_id, image=self.grid_image)