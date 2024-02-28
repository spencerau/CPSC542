import warnings
warnings.filterwarnings('ignore')

import os

# Run a shell command
#os.system('echo Hello, World!')
import shutil
from sklearn.model_selection import train_test_split


# # shell commmands to create dir and download dataset
# #!mkdir ~/.kaggle
# os.system('echo mkdir .kaggle')
# #!cp kaggle.json ~/.kaggle/
# os.system('echo cp kaggle.json /.kaggle/')
# #!chmod 600 ~/.kaggle/kaggle.json
# os.system('echo chmod 600 /.kaggle/kaggle.json')

# #!kaggle datasets download -d harishkumardatalab/food-image-classification-dataset
# os.system('echo kaggle datasets download -d harishkumardatalab/food-image-classification-dataset')
# #!unzip -q food-image-classification-dataset.zip -d food_image_classification
# os.system('echo unzip -q food-image-classification-dataset.zip -d food_image_classification')

# # Script to create Train/Test/Val Subdir with images
# # while still keeping labels as dir name

base_dir = 'Food Classification dataset'
classes_dir = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# Define new dataset structure
dataset_dir = 'food_image_classification/dataset'
train_dir = os.path.join(dataset_dir, 'train')
val_dir = os.path.join(dataset_dir, 'val')
test_dir = os.path.join(dataset_dir, 'test')

# Create new train, val, test directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Split data and move to respective directories
for cls in classes_dir:
    class_path = os.path.join(base_dir, cls)

    # Create class directories in train, val, and test
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

    # Get all images in the class directory
    all_files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]

    # Split the dataset into train+val and test sets
    train_val_files, test_files = train_test_split(all_files, test_size=0.10)
    # Split the train+val into train and validation sets
    train_files, val_files = train_test_split(train_val_files, test_size=0.111111)
    # makes sure its a 80% train, 10% test, 10% val split

    # Function to copy files to a new location
    def copy_files(files, src_dir, dest_dir):
        for f in files:
            shutil.copy(os.path.join(src_dir, f), os.path.join(dest_dir, f))

    # Copy the files to the new directories
    copy_files(train_files, class_path, os.path.join(train_dir, cls))
    copy_files(val_files, class_path, os.path.join(val_dir, cls))
    copy_files(test_files, class_path, os.path.join(test_dir, cls))