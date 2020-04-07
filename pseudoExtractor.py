import numpy as np
import pandas as pd
import tkinter, tkinter.filedialog 
import os
import platform



def get_file(read_mode=None):
    if platform.system() == 'Windows':
        #select file
        root = tkinter.Tk()
        if read_mode == None:
            file = tkinter.filedialog.askopenfile(parent=root, mode='rb', title='Choose a file')
        else:
            file = tkinter.filedialog.askopenfile(parent=root, mode=read_mode, title='Choose a file')

        if file != None:
            return file
    
    else:
        #manualy select file
        #list files
        list_of_files = os.listdir("./data")
        print("List of files in data directory")
        for i in range(len(list_of_files)):
            print((i + 1), ' ', list_of_files[i])

        choice = input("Select File ")
        if choice != None:
            fname = "./data/" + list_of_files[int(choice) - 1]
            return open(fname, 'r')
    #if no file selected end run
    raise Exception


def get_Hela():
    helaFile = get_file()
    dfControl = pd.read_csv(helaFile, sep=' ', header=None)
    print(dfControl)
    helaFile = get_file()
    dfModified = pd.read_csv(helaFile, sep=' ', header=None)
    print(dfModified)
    return dfControl, dfModified



