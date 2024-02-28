import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image

# Suppress warnings
warnings.filterwarnings('ignore')



def load_test_data(test_dir, image_size=(224, 224), batch_size=32):
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    
    return test_generator

def evaluate_model(model_path, test_generator):
    model = tf.keras.models.load_model(model_path)
    results = model.evaluate(test_generator)
    print(f"Test Loss: {results[0]:.4f}, Test Accuracy: {results[1]:.4f}")


def main():
    model_path = './a1_fine_tuned_model/'
    test_dir = 'food_image_classification/dataset/test'
    
    test_generator = load_test_data(test_dir)
    evaluate_model(model_path, test_generator)


main()

