import warnings
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')

base_dir = 'food_image_classification/dataset/train'

LABELS = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

#print("size: " + str(len(LABELS)))
#for i, label in enumerate(LABELS):
    #print(str(i) + ' ' + label)

def preprocess_image(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded_dims)

def predict_image_class(model, image_path):
    preprocessed_img = preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    #predicted_class_indices = np.argmax(predictions, axis=1)

    # Get indices of the top 3 predictions
    top_3_indices = np.argsort(predictions[0])[-3:][::-1]
    
    # Map indices to labels
    top_3_labels = [LABELS[index] for index in top_3_indices]
    
    return top_3_labels

model_path = './fine_tuned_model/'
model = load_model(model_path)

input_image_path = 'mcchicken.jpg'
predicted_labels = predict_image_class(model, input_image_path)

for i, label in enumerate(predicted_labels):
    print(f"Predicted Class {i}: " + f"{label}")



