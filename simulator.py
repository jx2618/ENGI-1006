import random
import math
from matplotlib import pyplot as plt
import numpy as np

def image_example():
    '''should produce red,purple,green squares
    on the diagonal, over a black background'''
    # RGB indexes
    red,green,blue = range(3)
    # img array 
    # all zeros = black pixels
    # shape: (150 rows, 150 cols, 3 colors)
    img = np.zeros((150,150,3))
    for x in range(50):
        for y in range(50):
            # red pixels
            img[x,y,red] = 1.0
            # purple pixels
            # set 3 color components 
            img[x+50, y+50,:] = (.5,.0,.5)
            # green pixels
            img[x+100,y+100,green] = 1.0
    plt.imshow(img)

def normpdf(x, mean, sd):
    """
    Return the value of the normal distribution 
    with the specified mean and standard deviation (sd) at
    position x.
    You do not have to understand how this function works exactly. 
    """
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom

def pdeath(x, mean, sd):
    start = x-0.5
    end = x+0.5
    step =0.01    
    integral = 0.0
    while start<=end:
        integral += step * (normpdf(start,mean,sd) + normpdf(start+step,mean,sd)) / 2
        start += step            
    return integral    
    
recovery_time = 4 # recovery time in time-steps
virality = 0.2    # probability that a neighbor cell is infected in 
                  # each time step                                                  

class Cell(object):

    def __init__(self ,x, y):
        self.x = x
        self.y = y 
        self.state = "S" # can be "S" (susceptible), "R" (resistant = dead), or 
                         # "I" (infected)
        self.time = 0
        
    def infect(self): # Step 2.1
        self.state = 'I'
        self.time = 0


    def process(self, adjacent_cells): # Step 2.3
        if self.state != 'I' or self.time == 0: # if cell is not infected or if time step is 0
            self.time += 1 # add 1 time step and move on
            return
        else:
            avoid_infection = random.random() # random float between 0 and 1
            for (x,y) in adjacent_cells:
                if avoid_infection <= virality: # if avoidance chance is less than virality, infect the neighbors
                    self.infect()

class Map(object):
    
    
    def __init__(self):
        self.height = 150
        self.width = 150           
        self.cells = {}

    def add_cell(self, cell): # Step 1.1 
        self.cells[(cell.x, cell.y)] = cell # appending to the dictionary
        
    def display(self): # Step 1.3
        image = np.zeros((150, 150, 3), dtype='float64')
        for ((x,y), cell) in self.cells.items(): # for dictionary key of (x,y) with value (cell) in cells
           if cell.state == 'S':
               image[x, y] = [0, 1, 0]
           elif cell.state == 'I':
               image[x,y] = [1,0,0]
           elif cell.state == 'R':
               image[x,y] = [.5,.5,.5]

        plt.imshow(image)  # display the map
    

    def adjacent_cells(self, x,y): # Step 2.2
        global adjacent_cells
        adjacent_cells = [] 
        north = (x, y+1)
        south = (x, y-1)
        east = (x-1, y)
        west = (x+1, y)
        if 0 < north[0] < 150 and 0 < north[1] < 151:
            adjacent_cells.append(north)
        if 0 < south[0] < 150 and 0 < south[1] < 151:
            adjacent_cells.append(south)
        if 0 < east[0] < 150 and 0 < east[1] < 151:
            adjacent_cells.append(east)
        if 0 < west[0] < 150 and 0 < west[1] < 151:
            adjacent_cells.append(west)
        return adjacent_cells
    
    def time_step(self):
        for ((x,y), cell) in self.cells.items():
            cell.time += 1
            cell.process(adjacent_cells)
        self.display()
    
            
def read_map(filename): # Step 1.2
    m = Map()
    with open(filename,'r') as file: # opening file
        for line in file:
            x, y = line.strip().split(",") # reading file for x, y coordinates
            x = int(x) # converting string to int
            y = int(y) 
            cell = Cell(x, y) # define cell as an instance of the Cell class with x, y attributes
            m.add_cell(cell) # calling the add_cell function on the newly defined cell
    return m

m = read_map('nyc_map.csv')
m.display()
m.cells[(39,82)].infect()
m.adjacent_cells(39,82)
m.time_step()
m.time_step()
