import warnings
import os
import pathlib
import hub

from tensorflow.keras.utils import Sequence
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, Dropout
from tensorflow.keras.models import Model

#from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Metric, MeanIoU

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Suppress warnings
warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


export_dir = './a2_base_new/'
gpu_device = '/device:GPU:1'
EPOCHS = 100
FT_EPOCHS = 50
PATIENCE = 10

# Steps to Use the Dataset

#   Explore JSON Files: Look into the dataset_meta.json and other JSON files to understand the metadata, like category definitions and dataset configuration.

#   Load Images and Masks: Write a data loader that loads images and their corresponding masks. 
#   Ensure that any transformation applied to the image is also applied to the mask to maintain alignment.

#   Adjust Data Generators: Adapt the data generators to use the images and masks. 

#   Create Segmentation Model: Choose a segmentation model like U-Net.
#   Construct the model so that it takes images as input and predicts segmentation masks.

#   Training: Train model using the loaded images and masks. Use appropriate loss functions and metrics for segmentation.

#   Evaluation: Use the test set to evaluate model's performance with segmentation-specific metrics like IoU.


class HubSegmentationDataGenerator(Sequence):

    num_classes = 1

    # Generates data for Keras from a Hub dataset
    def __init__(self, dataset, batch_size=32, image_size=(224, 224), shuffle=True):
        # Initialization
        self.dataset = dataset
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.dataset['images']) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        
        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.dataset['images']))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        # Generates data containing batch_size samples
        X = np.empty((self.batch_size, *self.image_size, 3), dtype=np.float32)
        y = np.empty((self.batch_size, *self.image_size, 1), dtype=np.float32)  # Target shape

        # Generate data
        for i, idx in enumerate(indexes):
            # Process the image (same as before)
            image = self.dataset['images'][int(idx)].numpy()
            image = cv2.resize(image, self.image_size)
            image = preprocess_input(image)

            # Process the mask
            mask = self.dataset['masks'][int(idx)].numpy()
             # Convert boolean mask to uint8 if it's boolean
            if mask.dtype == np.bool_:
                mask = mask.astype(np.uint8)  # Convert boolean to uint8

            if mask.ndim == 3 and mask.shape[-1] > 1:
                # Example: Taking the first channel if that's the relevant mask channel
                mask = mask[..., 0]

            mask = cv2.resize(mask, self.image_size)
            mask = np.squeeze(mask)  # Squeeze to remove any extra singleton dimensions
            
            # Ensure mask still has 3 dimensions (height, width, channel)
            if mask.ndim == 2:
                mask = np.expand_dims(mask, axis=-1)
            
            # Debug: print the shapes to verify correctness
            #print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")

            X[i,] = image
            y[i,] = mask

        return X, y


# Conv2D: These are convolutional layers that will process the input image.
# MaxPooling2D: This is used to reduce the spatial dimensions of the output from the previous layer.
# Conv2DTranspose: These are used for upsampling the feature maps.
# concatenate: This function is used to concatenate the feature maps from the downsampling path with the upsampled feature maps, 
# allowing the network to use information from multiple resolutions. This is the "skip connection" which is a key component of U-Net.


# # Convolution Block Function: perform two convolution operations
# # Create a convolutional block with two conv layers with ReLU activation.
# def conv_block(input_tensor, num_filters):
#     x = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(input_tensor)
#     x = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(x)
#     return x
    
# Adjusted Convolution Block Function with Dropout and Regularization
def conv_block(input_tensor, num_filters):
    x = Conv2D(num_filters, (3, 3), 
               activation='relu', 
               padding='same', 
               kernel_regularizer=l2(0.01))(input_tensor)
    
    x = Dropout(0.4)(x)

    x = Conv2D(num_filters, (3, 3), 
               activation='relu', 
               padding='same', 
               kernel_regularizer=l2(0.01))(x)
    return x


# Encoder Block Function: Applies the convolution block followed by max-pooling for downsampling.
# Create an encoder block with a conv block and max pooling.
def encoder_block(input_tensor, num_filters):
    x = conv_block(input_tensor, num_filters)
    p = MaxPooling2D((2, 2))(x)
    return x, p


# Create a decoder block with upsampling, concatenation and two conv layers.
def decoder_block(input_tensor, concat_tensor, num_filters):
    # # Upsample the feature map to increase its spatial dimensions
    #x = UpSampling2D((2, 2))(input_tensor)
    # Use a transposed convolution to increase spatial dimensions
    x = Conv2DTranspose(num_filters, kernel_size=(3, 3), strides=(2, 2), padding='same')(input_tensor)
    # Use skip connections to reintroduce detail by concatenating 
    # the upsampled features with the corresponding encoder features
    x = concatenate([x, concat_tensor])
    # Convolve the combined features to learn and integrate them for reconstruction
    x = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(x)
    return x


# Build and Compile U-Net Model
def build_and_compile_unet(input_shape, n_classes):

    inputs = Input(input_shape)

    # Encoder
    e1, p1 = encoder_block(inputs, 64)
    e2, p2 = encoder_block(p1, 128)
    e3, p3 = encoder_block(p2, 256)
    e4, p4 = encoder_block(p3, 512)
    
    # Bridge
    b = conv_block(p4, 1024)
    
    # Decoder
    d4 = decoder_block(b, e4, 512)
    d3 = decoder_block(d4, e3, 256)
    d2 = decoder_block(d3, e2, 128)
    d1 = decoder_block(d2, e1, 64)
    
    # Output
    # Adjusted for 498 classes
    #outputs = Conv2D(498, (1, 1), activation='softmax')(d1)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(d1)

    
    model = Model(inputs=[inputs], outputs=[outputs])
    
    # added Mean IoU as an additional metric
    model.compile(optimizer = Adam(1e-3), 
              loss = 'binary_crossentropy', 
              metrics = ['accuracy', MeanIoU(num_classes=2)])
    
    #print(model.summary())
    
    return model


def train_model(model, train_generator, validation_generator, epochs):
    # Early stopping callback to avoid overfitting and restore the weights with best metrics
    early_stopping = EarlyStopping(monitor='val_loss', 
                                   patience=PATIENCE, 
                                   verbose=1, 
                                   restore_best_weights=True)
    
    # Calculate steps per epoch for training and validation
    train_steps = len(train_generator)
    val_steps = len(validation_generator)

    history = model.fit(
        x=train_generator,
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=val_steps,
        callbacks=[early_stopping]
    )

    return history

def calculate_iou(generator, model):
    predictions = []
    true_masks = []
    for i in range(len(generator)):
        x, y = generator[i]
        preds = model.predict(x)
        preds = np.round(preds)  # Assuming binary segmentation
        predictions.extend(preds)
        true_masks.extend(y)

    # Flatten the arrays to compute IoU
    true_masks_flattened = np.concatenate(true_masks).flatten()
    predicted_masks_flattened = np.concatenate(predictions).flatten()

    iou_score = jaccard_score(true_masks_flattened, predicted_masks_flattened)
    return iou_score


def save_model(model, export_dir):
    # Save the TensorFlow model in a Keras-compatible format
    model.save(export_dir)

    converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
    tflite_model = converter.convert()
    tflite_model_file = pathlib.Path('./a2_finetune.tflite')
    tflite_model_file.write_bytes(tflite_model)


# Predict masks in smaller batches to avoid OOM errors.
def predict_masks_in_small_batches(model, images, batch_size=8):
    predicted_masks = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        preds = model.predict(batch)
        predicted_masks.extend(preds)
    return np.array(predicted_masks)


def predict_and_visualize_masks(model, generator, num_samples):
    images, true_masks = generator[0]  # Fetch the first batch
    predicted_masks = predict_masks_in_small_batches(model, images[:num_samples])

    save_dir = 'images'
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(num_samples):
        # Normalize the images to [0, 255] if they were preprocessed
        image_to_save = (images[i] * 255).astype('uint8')
        
        # Ensure masks are boolean or in the range [0, 1] if using sigmoid
        true_mask_to_save = true_masks[i].squeeze()  # Assuming true masks are already [0, 1]
        predicted_mask_to_save = (predicted_masks[i].squeeze() > 0.5).astype('uint8')  # Threshold sigmoid outputs

        # Save original image
        plt.imsave(os.path.join(save_dir, f"image_{i}.png"), image_to_save)
        
        # Save true mask
        plt.imsave(os.path.join(save_dir, f"true_mask_{i}.png"), true_mask_to_save, cmap='gray')
        
        # Save predicted mask
        plt.imsave(os.path.join(save_dir, f"predicted_mask_{i}.png"), predicted_mask_to_save, cmap='gray')


def main():
    # Directories
    ds_train = hub.dataset('hub/train')
    ds_val = hub.dataset('hub/val')
    ds_test = hub.dataset('hub/test')

    # Check if model already exists
    if not os.path.exists(export_dir):
        # Make sure to use GPU that is not being utilized; possibly need to fix this 
        with tf.device(gpu_device):
            train_generator = HubSegmentationDataGenerator(ds_train, batch_size=32, image_size=(224, 224), shuffle=True)
            validation_generator = HubSegmentationDataGenerator(ds_val, batch_size=32, image_size=(224, 224), shuffle=False)

            # Determine the number of classes
            n_classes = 1

            model = build_and_compile_unet(input_shape=(224, 224, 3), n_classes=n_classes)

            history = train_model(model, train_generator, validation_generator, epochs=100)

            save_model(model, export_dir)

            print("\nModel trained and saved.\n")

    else:

        print("\nModel already exists. Loading model for evaluation.\n")

        with tf.device(gpu_device):

            flag = False
            
            # Load the saved model
            model = tf.keras.models.load_model(export_dir)

            # preprocess and augment data
            train_generator = HubSegmentationDataGenerator(ds_train, batch_size=32, image_size=(224, 224), shuffle=True)
            validation_generator = HubSegmentationDataGenerator(ds_val, batch_size=32, image_size=(224, 224), shuffle=False)
            if flag:
                # Evaluate on validation data
                val_loss, val_accuracy = model.evaluate(validation_generator)
                print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")
                val_iou_manual = calculate_iou(validation_generator, model)
                print(f"Manual Validation IoU: {val_iou_manual}")
        
                # Evaluate on training data
                train_loss, train_accuracy = model.evaluate(train_generator)
                print(f"Training Loss: {train_loss}, Training Accuracy: {train_accuracy}")
                train_iou_manual = calculate_iou(train_generator, model)
                print(f"Manual Training IoU: {train_iou_manual}")

                # # Evaluate on validation data
                # val_loss, val_accuracy = model.evaluate(validation_generator)
                # print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")
                # val_iou_manual = calculate_iou(validation_generator, model)
                # print(f"Manual Validation IoU: {val_iou_manual}")

            else:
                predict_and_visualize_masks(model, validation_generator, 5)


main()
