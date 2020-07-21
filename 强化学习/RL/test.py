from tensorflow import keras

optimizer = keras.optimizers.Adam(0.1, clipvalue=1.0)
model = keras.models.Sequential()
model.compile(loss="mse", optimizer=optimizer)
keras.backend.set_value(model.optimizer.lr, 0.01)

a = 10
print(a//12)