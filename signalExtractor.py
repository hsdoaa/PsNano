import numpy as np
from statistics import mean
#pad data

def signal_data(x):
    max = 200
    paddedx = []
    paddedy = []
    #pad all elements to max length
    for i in range(len(x)):
        if len(x[i])>200:
             x[i]=x[i][:200]
        for j in range(len(x[i]), max):
              x[i].append(0)                

        paddedx.append(x[i])

    paddedx = np.array(paddedx)
    #reshape
    paddedx.reshape((paddedx.shape[0], max))
    return paddedx            
    













'''

        
	if len(x[i]) > 200:
	    len(x[i])=[:200]
        #f len(x[i]) < 200:
            #input raw signal data until signal array full
        for j in range(len(x[i]), max):
              x[i].append(0)
                #x[i].append(mean(x[i]))                  

            paddedx.append(x[i])
            paddedy.append(y[i])
        #elif len(x[i]) >= 200:
            #x[i].append("") #append nothing
            


    paddedx = np.array(paddedx)
    #reshape
    paddedx.reshape((paddedx.shape[0], max))
    return paddedx, paddedy             #Do we need to pad y if we pad x?

'''
