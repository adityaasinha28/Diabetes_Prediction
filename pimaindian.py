from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#fixing random seed 
seed=7;
np.random.seed(seed)

#load pima indians dataset
dataset=np.loadtxt("diabetes.csv",delimiter=",", skiprows=1)
#splitting into input X and output Y variables

X=dataset[:,0:8]
Y=dataset[:,8]

#creating the model
model=Sequential()
model.add(Dense(12,input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

#compiling the model

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

#fitting the model
model.fit(X,Y,epochs=300,batch_size=10)

#evaluting the model
scores=model.evaluate(X,Y)
print("%s: %.2f%%" %(model.metrics_names[1],scores[1]*100))

