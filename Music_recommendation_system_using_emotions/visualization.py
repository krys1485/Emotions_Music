# 1. Import Libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Input, Dense, Dropout, Conv2D,
                                     MaxPooling2D, Flatten, InputLayer)
from tensorflow.keras.optimizers import Adam

# 2. Load and Preprocess Data
label = []
dictionary = {}
X, y = [], []
c = 0

for file in os.listdir():
    if file.endswith(".npy") and not file.startswith("labels"):
        data = np.load(file)
        class_name = file.split('.')[0]

        X.append(data)
        y.append([class_name] * data.shape[0])

        if class_name not in dictionary:
            dictionary[class_name] = c
            label.append(class_name)
            c += 1

X = np.concatenate(X)
y = np.concatenate(y).reshape(-1, 1)

# Encode labels
y = np.vectorize(dictionary.get)(y.flatten())
y_cat = to_categorical(y)

# Reshape for CNN input
X = X.reshape(-1, 30, 34, 1)

# Shuffle & Split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# 3. Build the CNN Model
# Build Functional API Model
inputs = Input(shape=(30, 34, 1), name="input")
x = Conv2D(32, (3, 3), activation='relu', name='conv1')(inputs)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', name='conv2')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(y_cat.shape[1], activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 4. Train the Model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)

# 5. Plot Training History
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()

# 6. Evaluate the Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# 7. Confusion Matrix & Classification Report
y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(model.predict(X_test), axis=1)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=label, yticklabels=label, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_true, y_pred, target_names=label))

# 8. Visualize Filters (Weights of Conv Layers)
filters, biases = model.get_layer('conv1').get_weights()
print(f"Shape of filters: {filters.shape}")  # (3, 3, 1, 32)

fig, axs = plt.subplots(4, 8, figsize=(12, 6))
for i in range(32):
    f = filters[:, :, 0, i]
    ax = axs[i//8, i%8]
    ax.imshow(f, cmap='gray')
    ax.axis('off')
plt.suptitle("Conv1 Filters")
plt.show()

# 9. Visualize Activations (Feature Maps)
# Make sure model has been called at least once
_ = model.predict(X_test[:1])

# Build model to fetch intermediate outputs
activation_model = Model(
    inputs=model.input,
    outputs=[layer.output for layer in model.layers if 'conv' in layer.name]
)

# Get activations
sample = X_test[0].reshape(1, 30, 34, 1)
activations = activation_model.predict(sample)

# Visualize first conv layer feature maps
first_layer_activation = activations[0]
fig, axs = plt.subplots(4, 8, figsize=(12, 6))
for i in range(32):
    axs[i // 8, i % 8].imshow(first_layer_activation[0, :, :, i], cmap='viridis')
    axs[i // 8, i % 8].axis('off')
plt.suptitle("Feature Maps from Conv1")
plt.show()

# 10. Occlusion Sensitivity (Saliency-Like Map)
def occlusion_sensitivity(image, label_index, size=5):
    heatmap = np.zeros((image.shape[1], image.shape[2]))
    for i in range(0, image.shape[1] - size, size):
        for j in range(0, image.shape[2] - size, size):
            occluded = image.copy()
            occluded[0, i:i+size, j:j+size, :] = 0
            pred = model.predict(occluded)
            heatmap[i:i+size, j:j+size] = 1 - pred[0][label_index]
    return heatmap

true_class = np.argmax(y_test[0])
heatmap = occlusion_sensitivity(sample, true_class)

plt.imshow(heatmap, cmap='hot')
plt.title('Occlusion Sensitivity Map')
plt.colorbar()
plt.show()

# 11. Receptive Field (Conceptual)
print("\nReceptive Field Concept:")
print("- Conv1: kernel 3x3 → RF: 3x3")
print("- After MaxPool (2x2) → RF: 6x6")
print("- Conv2: 3x3 on pooled → RF: 10x10")
print("- MaxPool again → total RF ~ 20x20")
print("Deeper layers ‘see’ more of the input!")

# 12. Save Model & Labels
model.save("gesture_model.h5")
np.save("labels.npy", np.array(label))
