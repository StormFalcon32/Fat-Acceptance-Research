import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = data.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
X_train = X_train / 255
X_test = X_test / 255

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=4096)
_, test_acc = model.evaluate(X_test, y_test)

prediction = new_model.predict(X_test)
for i in range(10):
    plt.grid(False)
    plt.imshow(X_test[i], cmap='binary')
    plt.xlabel("Actual: " + class_names[y_test[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()
