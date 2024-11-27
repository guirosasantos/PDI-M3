import numpy as np
from keras.api.models import Sequential
from keras.api.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.api.datasets import cifar10
from keras.api.utils import to_categorical
from Lib import Sharpening, Upscaling, Print, Time, Filters

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
x_train = Filters.ApplyFiltersToImages(x_train)
x_test = Filters.ApplyFiltersToImages(x_test)

# Create the original model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # 10 classes
])

# Compile the original model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Save model
model.save('modelo-default-treinado-com-filtros.h5')

# Train the original model
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# Evaluate the original model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Acurácia no conjunto de teste: {test_accuracy * 100:.2f}%")

#? UNCOMMENT TO TEST UPSCALING, BUT COMMENT THE LINES 48-51
""" # Upsacle the images
x_train_upscaled = Upscaling.ApplyUpscaling(x_train)
x_test_upscaled = Upscaling.ApplyUpscaling(x_test) """

#! TESTS, UPSCALING TAKES TOO LONG, TESTING WITH ONLY 100 IMAGES
x_train_upscaled = Upscaling.ApplyUpscaling(x_train[:100])
x_test_upscaled = Upscaling.ApplyUpscaling(x_test[:100])
y_train_upscaled = y_train[:100]
y_test_upscaled = y_test[:100]

# Update the input shape for the model
input_shape = x_train_upscaled.shape[1:]

# Create the model for upscaled images
upscaled_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # 10 classes
])

# Compile the model for upscaled images
upscaled_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#? UNCOMMENT TO TEST UPSCALING, BUT COMMENT THE LINES 78-79
""" # Train the model on the upscaled dataset
upscaled_model.fit(x_train_upscaled, y_train, epochs=10, batch_size=64, validation_split=0.2)

# Evaluate the upscaled model
test_loss, test_accuracy = upscaled_model.evaluate(x_test_upscaled, y_test) """

#! TESTS, UPSCALING TAKES TOO LONG, TESTING WITH ONLY 100 IMAGES
upscaled_model.fit(x_train_upscaled, y_train_upscaled, epochs=10, batch_size=64, validation_split=0.2)
test_loss, test_accuracy = upscaled_model.evaluate(x_test_upscaled, y_test_upscaled)
print(f"Acurácia no conjunto de teste com upscaling: {test_accuracy * 100:.2f}%")

# Select 5 random images from the test set
original_first_img   = x_test[np.random.randint(0, x_test.shape[0])]
original_second_img  = x_test[np.random.randint(0, x_test.shape[0])]
original_third_img   = x_test[np.random.randint(0, x_test.shape[0])]
original_fourth_img  = x_test[np.random.randint(0, x_test.shape[0])]
original_fifth_img   = x_test[np.random.randint(0, x_test.shape[0])]

# Apply sharpening on the images
first_img   = Sharpening.ApplySharpening(original_first_img)
second_img  = Sharpening.ApplySharpening(original_second_img)
third_img   = Sharpening.ApplySharpening(original_third_img)
fourth_img  = Sharpening.ApplySharpening(original_fourth_img)
fifth_img   = Sharpening.ApplySharpening(original_fifth_img)

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
    original_first_img,
    first_img,
    first_img_upscaled,
    first_img_original_prediction_original_model,
    first_img_upscaled_prediction_upscalled_model,
    1)

Print.PrintResults(
    original_second_img,
    second_img,
    second_img_upscaled,
    second_img_original_prediction_original_model,
    second_img_upscaled_prediction_upscalled_model,
    2)

Print.PrintResults(
    original_third_img,
    third_img,
    third_img_upscaled,
    third_img_original_prediction_original_model,
    third_img_upscaled_prediction_upscalled_model,
    3)

Print.PrintResults(
    original_fourth_img,
    fourth_img,
    fourth_img_upscaled,
    fourth_img_original_prediction_original_model,
    fourth_img_upscaled_prediction_upscalled_model,
    4)

Print.PrintResults(
    original_fifth_img,
    fifth_img,
    fifth_img_upscaled,
    fifth_img_original_prediction_original_model,
    fifth_img_upscaled_prediction_upscalled_model,
    5)

# Stop the time count
elapsed_time = Time.StopTime(start_time)

# Print the elapsed time
Time.PrintElapsedTime(elapsed_time)
