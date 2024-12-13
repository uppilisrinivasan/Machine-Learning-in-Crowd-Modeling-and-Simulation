import sys
import tkinter
from tkinter import Button, Canvas, Menu

from scenario_elements import Scenario


class MainGUI():
    """
    Defines a simple graphical user interface.
    To start, use the `start_gui` method.
    """

    def __init__(self, scenarioDict):
        """"
        Saves the initial state of the simulation.
        """

        self.scenarioDict = scenarioDict

    def restart_scenario(self, canvas, canvas_image):
        """"
        Restarts the currently running scenario.
        """

        scenario = Scenario(
            self.scenarioDict['width'],
            self.scenarioDict['height'],
            self.scenarioDict['targets'],
            self.scenarioDict['obstacles'],
            self.scenarioDict['pedestrians'],
            self.scenarioDict['use_dijkstra']
        )

        # can be used to show pedestrians and targets
        scenario.to_image(canvas, canvas_image)

        # can be used to show the target grid instead
        # scenario.target_grid_to_image(canvas, canvas_image)

        self.stepBtn.configure(command=lambda: self.step_scenario(scenario, canvas, canvas_image))

    def step_scenario(self, scenario, canvas, canvas_image):
        """
        Moves the simulation forward by one step, and visualizes the result.

        Args:
            scenario (scenario_elements.Scenario): Add _description_
            canvas (tkinter.Canvas): Add _description_
            canvas_image (missing _type_): Add _description_
        """
        self.stepBtn['state'] = 'disabled'
        scenario.start_step()
        self.substep(scenario, canvas, canvas_image)

    def substep(self, scenario, canvas, canvas_image):
        if scenario.substep():
            scenario.to_image(canvas, canvas_image)
            canvas.after(200, lambda: self.substep(scenario, canvas, canvas_image))
        else:
            self.stepBtn['state'] = 'normal'

    def exit_gui(self, ):
        """
        Close the GUI.
        """
        sys.exit()

    def start_gui(self):
        """
        Creates and shows a simple user interface with a menu and multiple buttons.
        Only one button works at the moment: "step simulation".
        Also creates a rudimentary, fixed Scenario instance with three Pedestrian instances and multiple targets.
        """
        win = tkinter.Tk()
        win.geometry('500x500')  # setting the size of the window
        win.title('Cellular Automata GUI')

        canvas = Canvas(win, width=Scenario.GRID_SIZE[0], height=Scenario.GRID_SIZE[1])  # creating the canvas
        canvas_image = canvas.create_image(5, 50, image=None, anchor=tkinter.NW)
        canvas.pack()

        self.stepBtn = Button(win, text='Step simulation')
        self.stepBtn.place(x=20, y=10)
        btn = Button(win, text='Restart simulation', command=lambda: self.restart_scenario(canvas, canvas_image))
        btn.place(x=200, y=10)

        self.restart_scenario(canvas, canvas_image)

        win.mainloop()

        # menu = Menu(win)
        # win.config(menu=menu)
        # file_menu = Menu(menu)
        # menu.add_cascade(label='Simulation', menu=file_menu)
        # file_menu.add_command(label='New', command=self.create_scenario)
        # # file_menu.add_command(label='Restart', command=self.restart_scenario(canvas, canvas_image))
        # file_menu.add_command(label='Close', command=self.exit_gui)
