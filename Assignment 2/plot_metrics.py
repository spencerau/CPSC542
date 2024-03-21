import json
import matplotlib.pyplot as plt

# Load the training history
with open('training_history.json', 'r') as file:
    history = json.load(file)

# Create subplots
plt.figure(figsize=(21, 7))

# Plot training & validation accuracy values
plt.subplot(1, 3, 1)
plt.plot(history['accuracy'], label='Train')
plt.plot(history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 3, 2)
plt.plot(history['loss'], label='Train')
plt.plot(history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Plot training & validation IoU values
plt.subplot(1, 3, 3)
plt.plot(history['mean_io_u'], label='Train IoU') 
plt.plot(history['val_mean_io_u'], label='Validation IoU') 
plt.title('Model IoU')
plt.ylabel('IoU')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Save the figure
plt.savefig('training_metrics.png')
plt.close()
