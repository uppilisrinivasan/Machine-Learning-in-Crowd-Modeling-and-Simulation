import os
from pip import main
from prompt_toolkit import prompt
from gui import MainGUI
import json


def simulate():
    """"
    Starts a simulation instace.
    """

    name = prompt("scenario name:")
    path = "scenarios/" + name + '.json'
    gui : MainGUI

    try:
        with open(path, 'r') as scenarioFile:
            dic = json.load(scenarioFile)

            gui = MainGUI(dic)
    except:
        print("invalid file")
        return

    gui.start_gui()
        
def create():
    """"
    Creates a scenario and saves it.
    """

    name = prompt("scenario name: ")
    path = "scenarios/" + name + '.json'

    if os.path.exists(path):
        print('the name already exists')
    else:
        try:
            dic = {}
            dic['width'] = int(prompt("width (1, 1024): "))
            dic['height'] = int(prompt("height (1, 1024): "))
            
            no_pedestrians  = int(prompt("number of pedestrians: "))
            no_targets      = int(prompt("number of targets: "))
            no_obstacles    = int(prompt("number of obstacles: "))

            dic['pedestrians'] = []
            for i in range(no_pedestrians):
                print("pedestrian " + str(i))
                x = int(prompt('x: '))
                y = int(prompt('y: '))
                desired_distance = float(prompt('desired distance per timestep (1, 10): '))

                pedestrian = {}
                pedestrian['position'] = (x, y)
                pedestrian['desired_distance'] = desired_distance
                dic['pedestrians'].append(pedestrian)

            dic['targets'] = []
            for i in range(no_targets):
                print("target " + str(i))
                x = int(prompt('x: '))
                y = int(prompt('y: '))

                dic['targets'].append((x, y))

            dic['obstacles'] = []
            for i in range(no_obstacles):
                print("obstacle " + str(i))
                x = int(prompt('x: '))
                y = int(prompt('y: '))

                dic['obstacles'].append((x, y))

            dic['use_dijkstra'] = bool(prompt("For dijkstra algorithm, set True. For euclidean distance, set False: "))

            with open(path, 'w') as file:
                json.dump(dic, file)

        except:
            print('invalid input')

            
if __name__ == '__main__':
    while True:
        command = prompt("simulate, create or quit? (s/c/*):")

        if command == 's':
            simulate()
        elif command == 'c':
            create()
        else:
            quit()

            