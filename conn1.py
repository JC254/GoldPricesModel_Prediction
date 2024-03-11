import pandas as pd
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.layers.recurrent import LSTM
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


data=pd.read_csv('D:\Code\GoldMLmodel\merged_df.csv')

data = data.apply(lambda x: pd.to_numeric(x, errors='coerce') if x.name != 'DateTime' else x ).fillna(0)
data.set_index(['DateTime'],inplace=True)

x=data.drop(columns= list(data.columns[20:38]))
y=data.iloc[:,18]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

model = RandomForestRegressor()
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print(score)
y_pred = model.predict(x_test)
y_pred[:100]

print('This was sequential')

model = keras.Sequential()
model.add(Dense(units=256, input_shape=(11721,), activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1,activation='relu'))
model.compile(optimizer='sgd', loss='binary_crossentropy',metrics='accuracy')
model.fit(x_train, y_train, epochs=100,batch_size = 1000)
model.evaluate(x_test,y_test)

print('This was Random Forest')


model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(y_pred)

mse = mean_squared_error(y_test, y_pred)
print(mse)

print('This was GradientBoostingRegressor')


model = Sequential([
    Dense(64, activation='relu', input_shape=(11720,)),
    Dense(64, activation='relu'),
    Dense(1)  # Output layer with single neuron for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, mae = model.evaluate(x_test, y_test)
print("Mean Absolute Error:", mae)


print('This was FNN')