import numpy as np
from keras.api.models import Sequential
from keras.api.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, BatchNormalization, Dropout
import matplotlib.pyplot as plt
from keras.api.datasets import cifar10
from keras.api.utils import to_categorical
from keras.api.models import load_model
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from Lib import Sharpening, Upscaling, Print, Time

# Counts the execution time
start_time = Time.InitializeTimeCount()

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the images
x_train, x_test = x_train / 255.0, x_test / 255.0

# one-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Apply sharpening on the images
x_train = Sharpening.ApplySharpeningToImages(x_train)
x_test = Sharpening.ApplySharpeningToImages(x_test)

shape = x_train.shape[1:]

datagen = ImageDataGenerator(
    rotation_range=15,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(x_train)

# Create the original model
model = Sequential([
    Input(shape=shape),
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile the original model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the original model
history = model.fit(datagen.flow(x_train, y_train, batch_size=64),
                    epochs=50, validation_data=(x_test, y_test), verbose=1)

# Evaluate the original model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Acurácia no conjunto de teste: {test_accuracy * 100:.2f}%")

model.save('original_model.keras')

epochs_range = range(1, len(history.history['accuracy']) + 1)

Print.ShowTrainingResults(history, epochs_range)

# Upsacle the images
x_train_upscaled = Upscaling.ApplyUpscaling(x_train)
x_test_upscaled = Upscaling.ApplyUpscaling(x_test)

datagen_upscaled = ImageDataGenerator(
    rotation_range=15,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(x_train_upscaled)

# Update the input shape for the model
input_shape = x_train_upscaled.shape[1:]

# Create the model for upscaled images
upscaled_model = Sequential([
    Input(shape=input_shape),
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile the model for upscaled images
upscaled_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model on the upscaled dataset
history = upscaled_model.fit(datagen_upscaled.flow(x_train_upscaled, y_train, batch_size=64),
                     epochs=50, validation_data=(x_test_upscaled, y_test), verbose=1)

# Evaluate the upscaled model
test_loss, test_accuracy = upscaled_model.evaluate(x_test_upscaled, y_test)
print(f"Acurácia no conjunto de teste com upscaling: {test_accuracy * 100:.2f}%")

upscaled_model.save('upscaled_model.keras')

epochs_range = range(1, len(history.history['accuracy']) + 1)

Print.ShowTrainingResults(history, epochs_range)

# Select 5 random images from the test set
first_random_number   = np.random.randint(0, x_test.shape[0])
second_random_number  = np.random.randint(0, x_test.shape[0])
third_random_number   = np.random.randint(0, x_test.shape[0])
fourth_random_number  = np.random.randint(0, x_test.shape[0])
fifth_random_number   = np.random.randint(0, x_test.shape[0])

first_img   = x_test[first_random_number]
second_img  = x_test[second_random_number]
third_img   = x_test[third_random_number]
fourth_img  = x_test[fourth_random_number]
fifth_img   = x_test[fifth_random_number]

estimated_first_img   = y_test[first_random_number]
estimated_second_img  = y_test[second_random_number]
estimated_third_img   = y_test[third_random_number]
estimated_fourth_img  = y_test[fourth_random_number]
estimated_fifth_img   = y_test[fifth_random_number]

# Upscaling the images
first_img_upscaled  = Upscaling.ApplyUpscaling([first_img])
second_img_upscaled = Upscaling.ApplyUpscaling([second_img])
third_img_upscaled  = Upscaling.ApplyUpscaling([third_img])
fourth_img_upscaled = Upscaling.ApplyUpscaling([fourth_img])
fifth_img_upscaled  = Upscaling.ApplyUpscaling([fifth_img])

# Test results

# Original model, without upscaling
first_img_original_prediction_original_model  = model.predict(np.expand_dims(first_img, axis=0))
second_img_original_prediction_original_model = model.predict(np.expand_dims(second_img, axis=0))
third_img_original_prediction_original_model  = model.predict(np.expand_dims(third_img, axis=0))
fourth_img_original_prediction_original_model = model.predict(np.expand_dims(fourth_img, axis=0))
fifth_img_original_prediction_original_model  = model.predict(np.expand_dims(fifth_img, axis=0))

# Upscaled model, with upscaling
first_img_upscaled_prediction_upscalled_model = upscaled_model.predict(first_img_upscaled)
second_img_upscaled_prediction_upscalled_model = upscaled_model.predict(second_img_upscaled)
third_img_upscaled_prediction_upscalled_model = upscaled_model.predict(third_img_upscaled)
fourth_img_upscaled_prediction_upscalled_model = upscaled_model.predict(fourth_img_upscaled)
fifth_img_upscaled_prediction_upscalled_model = upscaled_model.predict(fifth_img_upscaled)

# Print the results without upscaling
Print.PrintResults(
    first_img,
    first_img_upscaled,
    first_img_original_prediction_original_model,
    first_img_upscaled_prediction_upscalled_model,
    estimated_first_img,
    1)

Print.PrintResults(
    second_img,
    second_img_upscaled,
    second_img_original_prediction_original_model,
    second_img_upscaled_prediction_upscalled_model,
    estimated_second_img,
    2)

Print.PrintResults(
    third_img,
    third_img_upscaled,
    third_img_original_prediction_original_model,
    third_img_upscaled_prediction_upscalled_model,
    estimated_third_img,
    3)

Print.PrintResults(
    fourth_img,
    fourth_img_upscaled,
    fourth_img_original_prediction_original_model,
    fourth_img_upscaled_prediction_upscalled_model,
    estimated_fourth_img,
    4)

Print.PrintResults(
    fifth_img,
    fifth_img_upscaled,
    fifth_img_original_prediction_original_model,
    fifth_img_upscaled_prediction_upscalled_model,
    estimated_fifth_img,
    5)

# Stop the time count
elapsed_time = Time.StopTime(start_time)

# Print the elapsed time
Time.PrintElapsedTime(elapsed_time)
