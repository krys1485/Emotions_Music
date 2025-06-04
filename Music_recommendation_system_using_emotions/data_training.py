import os  
import numpy as np 
import cv2 
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense 
from keras.models import Model

# Initialization
is_init = False
size = -1
label = []
dictionary = {}
c = 0

# Load all .npy files (except labels.npy) and assign labels
for i in os.listdir():
    if i.endswith(".npy") and not i.startswith("labels"):
        class_name = i.split('.')[0]
        data = np.load(i)

        if not is_init:
            is_init = True 
            X = data
            size = data.shape[0]
            y = np.array([class_name]*size).reshape(-1,1)
        else:
            X = np.concatenate((X, data))
            y = np.concatenate((y, np.array([class_name]*data.shape[0]).reshape(-1,1)))

        label.append(class_name)
        dictionary[class_name] = c  
        c += 1

# Convert class names to numeric labels
for i in range(y.shape[0]):
    y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")

# One-hot encode the labels
y = to_categorical(y)

# Shuffle data
X_new = np.empty_like(X)
y_new = np.empty_like(y)
cnt = np.arange(X.shape[0])
np.random.shuffle(cnt)
for idx, i in enumerate(cnt):
    X_new[idx] = X[i]
    y_new[idx] = y[i]

# Define the model
ip = Input(shape=(X.shape[1],))  # ✅ FIXED: Tuple shape
m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)
op = Dense(y.shape[1], activation="softmax")(m) 

model = Model(inputs=ip, outputs=op)
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

# Train the model
model.fit(X_new, y_new, epochs=50)

# Save model and label mapping
model.save("model.h5")
np.save("labels.npy", np.array(label))