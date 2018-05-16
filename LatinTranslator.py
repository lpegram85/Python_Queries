import tkinter


class ShowInfoGUI:
    def __init__(self):
        # Create the main window
        self.main_window = tkinter.Tk()

        # Create two frames
        self.top_frame = tkinter.Frame(self.main_window)
        self.bottom_frame = tkinter.Frame(self.main_window)

        # Create a blank label in the top frame
        self.value = tkinter.StringVar()
        self.value.set('Latin Translator')

        self.address_label = tkinter.Label(self.top_frame,textvariable=self.value)

        # Create the two buttons in the bottom frame
        self.sinister_button = tkinter.Button(self.bottom_frame,
                                             text='Sinister', command=self.show_info1)
        self.dexter_button = tkinter.Button(self.bottom_frame,
                                             text='Dexter', command=self.show_info2)
        self.medium_button = tkinter.Button(self.bottom_frame,
                                             text='Medium', command=self.show_info3)

        self.quit_button = tkinter.Button(self.bottom_frame,
                                          text='Quit', command=self.main_window.destroy)

        # Pack the label
        self.address_label.pack()
        # Pack the buttons
        self.sinister_button.pack(side='left')
        self.dexter_button.pack(side='left')
        self.medium_button.pack(side='left')
        self.quit_button.pack(side='left')

        # Pack the frames
        self.top_frame.pack()
        self.bottom_frame.pack()

        # Enter the tkinter main loop
        tkinter.mainloop()

    # Define the show_info function
    def show_info1(self):
        self.value.set('Left')

    def show_info2(self):
        self.value.set('Right')


    def show_info3(self):
       self.value.set('Center')
# Create an instance of ShowInfoGUI
show_info = ShowInfoGUI()
