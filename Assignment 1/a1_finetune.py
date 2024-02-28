import warnings
import os
import pathlib

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Corrected import for VGG16 and preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall

from tensorflow.keras.callbacks import EarlyStopping

# Suppress warnings
warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3, 4, 5, 6, 7, 8"



export_dir = './a1_base/'
fine_tuned_export_dir = './a1_fine_tuned_model/'
gpu_device = '/device:GPU:7'
EPOCHS = 200
FT_EPOCHS = 100
PATIENCE = 15

# Steps to Using a Pre-Trained Model

# Load a Pre-trained Model: Load VGG16 (or any other pre-trained model) with its pre-trained weights, ready to be adapted to a new task.

# Freeze the Weights: Freeze the weights of the pre-trained layers by setting requires_grad to False. This prevents the weights from being updated 
# during training, which is crucial for retaining the learned features

# Modify the Classifier: Replace the last part of the classifier with custom layers tailored to the new task. This typically involves adding new fully 
# connected layers, activation functions, dropout layers for regularization, and adjusting the output layer to match the number of classes in the new task.

# Train Only the Custom Layers: Since the pre-trained layers' weights are frozen, only the weights of the newly added layers are updated during training. 
# This allows the model to apply the generic features learned from the original dataset (ImageNet) to the new task.


def setup_data_generators(train_dir, val_dir, test_dir, batch_size=32):
    # Only augment training data with various image manipulation operations
    train_datagen = ImageDataGenerator(
        # Use VGG16's preprocess_input
        preprocessing_function=preprocess_input,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    # The preprocess_input function standardizes images in a way that matches the original preprocessing applied to the images when the VGG16 model was trained. 
    # This includes color channel normalization based on the means of the ImageNet dataset, which is crucial for models pre-trained on ImageNet.
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, validation_generator, test_generator


def build_and_compile_model(n_classes):
    # Load VGG16 pre-trained on ImageNet without the top layer (classifier)
    base_model = VGG16(weights='imagenet', 
                       include_top=False, 
                       input_shape=(224, 224, 3))
    
    # Freeze the convolutional base
    base_model.trainable = False

    # Create a custom classifier as the new top layer
    # dropout: increase to lower overfitting, decrease for better learning 
    # 0.2 = 20% of the input units are randomly excluded from each update cycle.
    classifier = Sequential([
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        Dense(n_classes, activation='softmax')
    ])
    # Stack the base model and the custom classifier
    model = Sequential([
        base_model, 
        classifier
    ])
    # Compile the model
    # sets the learning rate for the optimizer to 0.001 = 1e-3
    model.compile(optimizer = Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(model, train_generator, validation_generator, epochs=EPOCHS):
    # Early stopping callback to avoid overfitting and restore the weights with best metrics
    early_stopping = EarlyStopping(monitor='val_loss', 
                                   patience=PATIENCE, 
                                   verbose=1, 
                                   restore_best_weights=True)

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        callbacks=[early_stopping]
    )
    return history


def save_model(model, export_dir):
    # Save the TensorFlow model in a Keras-compatible format
    model.save(export_dir)

    converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
    tflite_model = converter.convert()
    tflite_model_file = pathlib.Path('./a1_finetune.tflite')
    tflite_model_file.write_bytes(tflite_model)


def unfreeze_model_layers(model, num_layers_to_unfreeze):
    # Unfreeze the top `num_layers_to_unfreeze` layers of the model
    for layer in model.layers[-num_layers_to_unfreeze:]:
        layer.trainable = True


def fine_tune_model(model, train_generator, validation_generator, initial_epochs=EPOCHS, fine_tune_epochs=FT_EPOCHS):
    # Re-compile the model with a lower learning rate
    # sets the learning rate for the optimizer to 0.0001 = 1e-4
    # switch to Adam
    model.compile(optimizer = Adam(learning_rate=1e-4), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # Early stopping callback so i don't shit up the DGX Server plus prevent overfitting and training on excess noise
    early_stopping = EarlyStopping(monitor='val_loss', 
                                   patience=PATIENCE, 
                                   verbose=1, 
                                   restore_best_weights=True)

    # Continue training
    total_epochs = initial_epochs + fine_tune_epochs
    history_fine = model.fit(train_generator,
                             steps_per_epoch=train_generator.samples // train_generator.batch_size,
                             epochs=total_epochs,
                             initial_epoch=initial_epochs,  # Continue from previous training
                             validation_data=validation_generator,
                             validation_steps=validation_generator.samples // validation_generator.batch_size,
                             callbacks=[early_stopping])

    return history_fine


def main():
    # Directories
    train_dir = 'food_image_classification/dataset/train'
    val_dir = 'food_image_classification/dataset/val'
    test_dir = 'food_image_classification/dataset/test'

    # Check if model already exists
    if not os.path.exists(export_dir):
        # Make sure to use GPU that is not being utilized; possibly need to fix this 
        with tf.device(gpu_device):
            train_generator, validation_generator, _ = setup_data_generators(train_dir, val_dir, test_dir)

            # Determine the number of classes
            n_classes = len(train_generator.class_indices)

            model = build_and_compile_model(n_classes)

            history = train_model(model, train_generator, validation_generator)

            save_model(model, export_dir)

            print("\nModel trained and saved.\n")

    else:

        print("\nModel already exists. Loading model for fine-tuning or evaluation.\n")

        with tf.device(gpu_device):
            # load in model
            if not os.path.exists(fine_tuned_export_dir):
                print("\nFirst Time Fine Tuning; Unfreezing 3 Layers\n")
                model = tf.keras.models.load_model(export_dir)
                layers_unfreeze = 3

            elif os.path.exists(fine_tuned_export_dir):
                print("\nSequential Time Fine Tuning: Unfreezing Additional Layers\n")
                model = tf.keras.models.load_model(fine_tuned_export_dir)
                EPOCHS += FT_EPOCHS
                FT_EPOCHS = 50
                layers_unfreeze = 5

            # preprocess and augment data
            train_generator, validation_generator, _ = setup_data_generators(train_dir, val_dir, test_dir)
        
            # unfreeze layers for fine tuning
            # try for 3, 5, etc
            unfreeze_model_layers(model, num_layers_to_unfreeze = layers_unfreeze)
            
            # Fine-tune the model
            fine_tune_history = fine_tune_model(model, train_generator, validation_generator)

            # Call the save_model function to save the fine-tuned model
            save_model(model, fine_tuned_export_dir)


main()
