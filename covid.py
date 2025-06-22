# -*- coding: utf-8 -*-

# 1) IMPORTING LIBRARIES
import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle

# 2) CLONING DATASET & SETTING PATHS
main_dir = "tri_zip/tri/Chext-X-ray-Images-Data-Set"

# SETTING TRAIN AND TEST DIRECTORY
train_dir = os.path.join(main_dir, "DataSet/Data/train")
test_dir = os.path.join(main_dir, "DataSet/Data/test")

# SETTING DIRECTORY FOR COVID AND NORMAL IMAGES
train_covid_dir = os.path.join(train_dir, "COVID19")
train_normal_dir = os.path.join(train_dir, "NORMAL")

test_covid_dir = os.path.join(test_dir, "COVID19")
test_normal_dir = os.path.join(test_dir, "NORMAL")

# MAKING SEPARATE FILE LISTS
train_covid_names = os.listdir(train_covid_dir)
train_normal_names = os.listdir(train_normal_dir)

test_covid_names = os.listdir(test_covid_dir)
test_normal_names = os.listdir(test_normal_dir)

# 3) DATA VISUALIZATION
import matplotlib.image as mpimg

rows = 4
columns = 4

fig = plt.gcf()
fig.set_size_inches(12, 12)

covid_img = [os.path.join(train_covid_dir, filename) for filename in train_covid_names[0:8]]
normal_img = [os.path.join(train_normal_dir, filename) for filename in train_normal_names[0:8]]

merged_img = covid_img + normal_img

for i, img_path in enumerate(merged_img):
    title = img_path.split("/")[-2]  # Extract class name
    plot = plt.subplot(rows, columns, i + 1)
    plot.axis("Off")
    img = mpimg.imread(img_path)
    plot.set_title(title, fontsize=11)
    plt.imshow(img, cmap="gray")

plt.show()

# 4) DATA PREPROCESSING AND AUGMENTATION
dgen_train = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255,
                                validation_split=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True)

dgen_validation = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

dgen_test = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

train_generator = dgen_train.flow_from_directory(train_dir,
                                                 target_size=(150, 150),
                                                 subset='training',
                                                 batch_size=32,
                                                 class_mode='binary')

validation_generator = dgen_train.flow_from_directory(train_dir,
                                                      target_size=(150, 150),
                                                      subset="validation",
                                                      batch_size=32,
                                                      class_mode="binary")

test_generator = dgen_test.flow_from_directory(test_dir,
                                               target_size=(150, 150),
                                               batch_size=32,
                                               class_mode="binary")

print("Class Labels are:", train_generator.class_indices)
print("Image shape is:", train_generator.image_shape)

# 5) BUILDING CONVOLUTIONAL NEURAL NETWORK MODEL
model = tf.keras.models.Sequential()

# 1) CONVOLUTIONAL LAYER - 1
model.add(tf.keras.layers.Conv2D(32, (5, 5), padding="same", activation="relu", input_shape=train_generator.image_shape))

# 2) POOLING LAYER - 1
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

# 3) DROPOUT LAYER - 1
model.add(tf.keras.layers.Dropout(0.5))

# 4) CONVOLUTIONAL LAYER - 2
model.add(tf.keras.layers.Conv2D(64, (5, 5), padding="same", activation="relu"))

# 5) POOLING LAYER - 2
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

# 6) DROPOUT LAYER - 2
model.add(tf.keras.layers.Dropout(0.5))

# 7) FLATTENING LAYER
model.add(tf.keras.layers.Flatten())

# 8) DENSE LAYER
model.add(tf.keras.layers.Dense(256, activation='relu'))

# 9) DROPOUT LAYER - 3
model.add(tf.keras.layers.Dropout(0.5))

# 10) OUTPUT LAYER
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# PRINTING MODEL SUMMARY
model.summary()

# 6) COMPILING AND TRAINING THE MODEL
model.compile(tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator,
                    epochs=2,
                    validation_data=validation_generator)

# 7) PERFORMING EVALUATION

# PLOT GRAPH BETWEEN TRAINING AND VALIDATION LOSS
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title("Training and validation losses")
plt.xlabel('epoch')

# PLOT GRAPH BETWEEN TRAINING AND VALIDATION ACCURACY
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Training', 'Validation'])
plt.title("Training and validation accuracy")
plt.xlabel('epoch')

# GETTING TEST ACCURACY AND LOSS
test_loss, test_acc = model.evaluate(test_generator)
print("Test Set Loss:", test_loss)
print("Test Set Accuracy:", test_acc)

# # 8) PREDICTION ON NEW DATA (UPLOAD FILES)


from tkinter import Tk, filedialog

# Load the trained model
# model = tf.keras.models.load_model("model.h5")

# Open a file dialog to select an image
Tk().withdraw()  # Hide the root Tkinter window
img_path = filedialog.askopenfilename(title="Select an X-ray Image",
                                      filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])

# Check if a file was selected
if img_path:
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(150, 150))
    images = tf.keras.preprocessing.image.img_to_array(img)
    images = np.expand_dims(images, axis=0)

    # Make a prediction
    prediction = model.predict(images)

    # Display result
    if prediction < 0.5:
        print("The report is **COVID-19 Positive**")
    else:
        print("The report is **COVID-19 Negative**")
else:
    print("No file selected.")

# 9) SAVING THE MODEL PROPERLY

model.save("model.h5")
pickle.dump(model, open("model3.pkl", "wb"))
