# Human Action Recognition (HAR)  Project
## Project Overview
This project focuses on Human Action Recognition using the HAR dataset from Kaggle. The goal is to develop models capable of accurately recognizing various human actions from video data.

## Dataset
The dataset used for this project is the Human Action Recognition (HAR) dataset, which can be found here <a id='ssFeatures'>https://www.kaggle.com/datasets/meetnagadia/human-action-recognition-har-dataset</a>

## Project Breakdown
### 1. Data Analysis (Visualization, Data Cleaning)
Visualization: Utilized various plotting techniques to visualize the distribution of actions and understand the dataset better.
Data Cleaning: Handled missing values, removed duplicates, and corrected any inconsistencies in the dataset.
### 2. Feature Extraction (if required)
Extracted relevant features to improve model performance, if necessary.
### 3. Data Preprocessing (Data Preparation)
Resized images to a consistent size of 224x224.
Normalized pixel values for better model convergence.
Split the dataset into training, validation, and testing sets.
### 4. Callbacks
ModelCheckpoint: Saved the best model based on validation accuracy during training.
EarlyStopping: Stopped training when validation accuracy stopped improving to avoid overfitting.
### 5. Distributed Strategies (optional)
Implemented distributed training strategies to speed up the training process, if necessary.
### 6. Training and Validation
## Trained two models:
#### Custom CNN Model:
```
def build_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_model((224, 224, 3), 15)
```
#### Pretrained VGG16 Model:
```
# Initialize a Sequential model
cnn_model = Sequential()

# Adding an InputLayer to the Sequential model
cnn_model.add(InputLayer(shape=(224, 224, 3)))

# Initialize VGG16 without the top layers
pretrained_model = tf.keras.applications.VGG16(include_top=False,
                                               input_shape=(224, 224, 3),
                                               pooling='avg',
                                               weights='imagenet')

# Freezing the deeper layers
for layer in pretrained_model.layers:
    layer.trainable = False

# Adding the pretrained VGG16 model to the Sequential model
cnn_model.add(pretrained_model)

# Adding the rest of the layers to the model
cnn_model.add(Flatten())
cnn_model.add(Dense(512, activation='relu'))
cnn_model.add(Dense(15, activation='softmax'))

# Compiling the model
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```
### 7. Testing
Evaluated the models on the test set to assess their performance.

### Conclusion
This project demonstrates the application of deep learning techniques to the task of Human Action Recognition. The results highlight the effectiveness of both custom CNN and pretrained VGG16 models in accurately recognizing human actions from video data. The system can be further improved and adapted for various real-world applications.

## Contact
For any questions or suggestions, feel free to reach out!
