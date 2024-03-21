import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model

# Load the image
img_path = 'best_worst_results/worst/img_72_22.png'
img = load_img(img_path, target_size=(224, 224))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Prepare the Grad-CAM model
model = tf.keras.models.load_model('a2_base') 
#model.summary()

last_conv_layer_name = 'conv2d_18'
grad_model = Model(inputs=model.inputs, outputs=[model.get_layer(last_conv_layer_name).output, model.output])

# Generate the Grad-CAM heatmap
with tf.GradientTape() as tape:
    last_conv_layer_output, preds = grad_model(img_array)
    class_channel = preds[:, 0]  # Assuming binary classification

# This is the gradient of the output neuron (where the class is predicted) with regard to the output feature map of the last conv layer
grads = tape.gradient(class_channel, last_conv_layer_output)

# Vector of mean intensity of the gradients over the specific feature map channel
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

# We multiply each channel in the feature map array by the corresponding gradient's mean intensity
last_conv_layer_output = last_conv_layer_output[0]
heatmap = tf.reduce_sum(tf.multiply(pooled_grads, last_conv_layer_output), axis=-1)

# ReLU activation and normalization
heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
heatmap = heatmap.numpy()

# Resize heatmap to original image size
heatmap = cv2.resize(heatmap, (img.size[1], img.size[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# Superimpose the heatmap on original image
superimposed_img = heatmap * 0.4 + img
cv2.imwrite('best_worst_results/best/gradcam_img_8_4.png', superimposed_img)

# Display GradCam
plt.axis('off')
plt.imshow(superimposed_img / 255) 
plt.savefig('gradcam2.png', bbox_inches='tight', pad_inches=0)
plt.close() 

