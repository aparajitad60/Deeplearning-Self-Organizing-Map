# Self Organizing Maps

#imposting libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#imposting dataset
dataset = pd.read_csv('Credit_Card_Applications.csv') #y = yes or no the application was approved
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scalling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)

#training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5) #we dont have a lot of data, so we will make 10 by 10 grid, as it should not be small we want outliers pretty evident, hence x=10, y=10; input len = no of features in X dataset; sigma= the radius of the diff neigborhood of the grid; l_r= weight of updation in each iteration
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100) #no of iterations repeat from step 4 to 9(in the steps)

#visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T) #inside pcolor fn,add all the MID(mean Interneuron distances) for all the winning nodes of our SOM, to get mean=Distance_map()
#distance_map() will return all MID'in one matrix for all the winning nodes, .T = transpose
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x) #getting winning node for customer x
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

#finding the frauds
mappings = som.win_map(X)
desfrauds = mappings[(6,3)]
frauds = sc.inverse_transform(frauds)
#mappings = som.win_map(X)
#frauds = np.concatenate((mappings[(8,1)], ma), axis = 0)














