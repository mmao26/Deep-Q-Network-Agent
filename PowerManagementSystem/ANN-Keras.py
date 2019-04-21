
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(0)

# Input array
# Example: 5 groups of data. Each group of data has 4 input features. 
X = np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1],[0,0,1,1],[1,1,1,1]])

# Output array
# Example: 5 groups of data. Each group of data has 1 output. 
#y = np.array([[1,0],[1,1],[0,0],[0,1],[1,1]])
y = np.array([1,1,0,0,1])

# create model
model = Sequential()
model.add(Dense(2, input_dim=4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='linear'))
# Compile model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])


# Fit the model
model.fit(X, y, epochs=2000, batch_size=3, verbose=0)
w = model.get_weights()
print (w)

# evaluate the model
scores = model.evaluate(X, y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
