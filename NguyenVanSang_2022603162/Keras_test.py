import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("/content/outputs", exist_ok=True)

# 1) Load & preprocess
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32")/255.0
x_test  = x_test.astype("float32")/255.0
x_train = np.expand_dims(x_train, -1)  # (N,28,28,1)
x_test  = np.expand_dims(x_test, -1)

# 2) Build model
model = keras.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3) Train
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    callbacks=[
        keras.callbacks.ModelCheckpoint(
            "/content/outputs/mnist_cnn_best.keras",
            monitor="val_accuracy",
            save_best_only=True
        )
    ]
)

# 4) Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("Test accuracy:", test_acc)

# 5) Plot accuracy/loss & SAVE figures
plt.figure()
plt.plot(history.history["accuracy"], label="acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.title("Accuracy")
plt.xlabel("epoch"); plt.ylabel("acc")
plt.legend()
plt.savefig("/content/outputs/mnist_accuracy.png", dpi=150)
plt.show()

plt.figure()
plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("Loss")
plt.xlabel("epoch"); plt.ylabel("loss")
plt.legend()
plt.savefig("/content/outputs/mnist_loss.png", dpi=150)
plt.show()

# 6) Predict a few samples & SAVE grid image
preds = model.predict(x_test[:9])
import matplotlib.pyplot as plt
plt.figure(figsize=(6,6))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[i].squeeze(), cmap="gray")
    plt.title(int(np.argmax(preds[i])))
    plt.axis("off")
plt.tight_layout()
plt.savefig("/content/outputs/mnist_preds.png", dpi=150)
plt.show()

# 7) Save final model
model.save("/content/outputs/mnist_cnn_final.keras")
print("Saved to /content/outputs/")
