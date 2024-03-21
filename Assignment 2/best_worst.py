import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import jaccard_score
from tensorflow.keras.metrics import Metric, MeanIoU
import os
import hub
from a2_train import HubSegmentationDataGenerator


model = tf.keras.models.load_model('a2_base_new')
ds_val = hub.dataset('hub/val')
validation_generator = HubSegmentationDataGenerator(ds_val, batch_size=32, image_size=(224, 224), shuffle=False)

def compute_and_save_best_worst_results(model, generator, num_samples, save_dir='best_worst_results'):
    """Compute IoU scores, determine best and worst results, and save them."""
    ious = compute_iou_per_image(generator, model, num_samples)
    sorted_indices = np.argsort(ious)
    worst_indices = sorted_indices[:3]
    best_indices = sorted_indices[-3:]

    save_images_and_masks(generator, best_indices, os.path.join(save_dir, 'best'))
    save_images_and_masks(generator, worst_indices, os.path.join(save_dir, 'worst'))


def compute_iou_per_image(generator, model, num_samples):
    """Compute IoU for each image and return the IoU scores."""
    ious = []
    for i in range(num_samples):
        x, y_true = generator[i]
        y_pred = model.predict(x)
        y_pred_thresholded = (y_pred > 0.5).astype(np.int32)
        
        # Assuming binary segmentation and IoU is computed per image
        for j in range(len(y_true)):
            y_true_j = y_true[j].astype(np.int32)
            y_pred_j = y_pred_thresholded[j]
            iou = MeanIoU(num_classes=2)
            iou.update_state(y_true_j.flatten(), y_pred_j.flatten())
            ious.append(iou.result().numpy())
    return np.array(ious)

def normalize_image(image):
    """Normalize image data to [0, 1] range."""
    # Clip values to ensure they fall within [0, 255]
    image_clipped = np.clip(image, 0, 255)
    # Normalize to [0, 1]
    return image_clipped / 255.0


def save_images_and_masks(generator, indices, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    for idx in indices:
        x, y_true = generator[idx]
        y_pred = model.predict(x)

        for i in range(len(x)):  # Assuming `x` is a batch of images.
            img = x[i]
            true_mask = y_true[i].squeeze()
            pred_mask = y_pred[i].squeeze()

            # Normalize the image if necessary
            img_normalized = normalize_image(img)

            # Ensure masks are 2D
            if true_mask.ndim == 3 and true_mask.shape[-1] == 1:
                true_mask = true_mask[..., 0]
            if pred_mask.ndim == 3 and pred_mask.shape[-1] == 1:
                pred_mask = pred_mask[..., 0]

            # Assuming binary masks, threshold predictions to get binary output
            pred_mask_binary = (y_pred[i].squeeze() > 0.5).astype(np.float32)
            
            # Convert true mask to binary if it's not already
            true_mask_binary = (y_true[i].squeeze() > 0.5).astype(np.float32)

            # Save images
            plt.imsave(f'{save_dir}/img_{idx}_{i}.png', img_normalized, cmap='gray')
            plt.imsave(f'{save_dir}/true_mask_{idx}_{i}.png', true_mask_binary, cmap='gray')
            plt.imsave(f'{save_dir}/pred_mask_{idx}_{i}.png', pred_mask_binary, cmap='gray')




def main():
    compute_and_save_best_worst_results(model, validation_generator, 3)

if __name__ == "__main__":
    main()

