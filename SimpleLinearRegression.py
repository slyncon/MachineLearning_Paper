# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 14:48:24 2017
@author: Lyncon Rodrigo, Caio Vinicius, Felipe Vieira
"""

# Simple Linear Regression

# Importar as libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the datasets

dataset = pd.read_csv('classesxexperience.csv')
# independent variable 
X = dataset.iloc[:, :-1].values
# comma to get all the columns
# -1 except for the last column

# dependent variable
y = dataset.iloc[:, 1].values
# data formatted

# Split the dataset in Training set e Test set

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Prepare Simple Linear Regression for the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Preview the results from the testing set
y_pred = regressor.predict(X_test)

# Visualize training set

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train,regressor.predict(X_train), color = 'blue')
plt.title('Curso vs Experiencia na Area (Training set)')
plt.xlabel('Meses de Curso')
plt.ylabel('Meses de Experiencia')
plt.show()

# Visualize results from Test set

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train,regressor.predict(X_train), color = 'purple')
plt.title('Curso vs Experiencia na Area(Test set)')
plt.xlabel('Meses de Curso')
plt.ylabel('Meses de Experiencia')
plt.show()
