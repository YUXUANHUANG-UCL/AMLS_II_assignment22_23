import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_file_list(hires_folder, lowres_folder):
    """Get a dataframe with pairs of filenames, one from a high resolution folder and the other from a low resolution folder.

    Args:
        hires_folder (str): Path to folder containing high resolution images.
        lowres_folder (str): Path to folder containing low resolution images.

    Returns:
        pd.DataFrame: A dataframe with pairs of filenames, one from the high resolution folder and the other from the low resolution folder.
    """
    # Create a dataframe with two columns: low_res and high_res
    file_list = pd.DataFrame({'low_res': os.listdir(lowres_folder), 'high_res': os.listdir(hires_folder)})

    # Sort the filenames in alphabetical order
    file_list['low_res'] = sorted(os.listdir(lowres_folder))
    file_list['high_res'] = sorted(os.listdir(hires_folder))

    # Create the full path of the low_res and high_res filenames using their respective folders
    file_list['low_res'] = file_list['low_res'].apply(lambda x: os.path.join(lowres_folder,x))
    file_list['high_res'] = file_list['high_res'].apply(lambda x: os.path.join(hires_folder,x))

    return file_list
  
def imageGenerator(train_generator):
    """
    A generator that yields batches of low-resolution and high-resolution images.

    Args:
    train_generator (tensorflow.keras.preprocessing.image.DirectoryIterator): 
        A directory iterator that generates batches of images.

    Yields:
    tuple: A tuple of low-resolution and high-resolution images.

    """
    for (low_res, hi_res) in train_generator:
        yield (low_res, hi_res)

def train_val_generator(df_type, sample_type, batch_size):
    """Generates image data for training, validation and testing sets.

    Args:
        df_type (str): Type of data frame used for training ('balanced' or 'imbalanced').
        sample_type (str): Type of sampling used for training ('random' or 'smote').
        batch_size (int): Number of images per batch.

    Returns:
        tuple: A tuple containing:
            - int: Number of samples in the training set.
            - int: Number of samples in the validation set.
            - int: Number of samples in the testing set.
            - generator: Generator object for the training set images.
            - generator: Generator object for the validation set images.
            - generator: Generator object for the testing set images.
    """
    base_directory = 'Datasets'
    train_hires_folder = os.path.join(base_directory, 'train_hr')
    train_lowres_folder = os.path.join(base_directory, 'train_' + df_type + '_' + sample_type)
    val_hires_folder = os.path.join(base_directory, 'val_hr')
    val_lowres_folder = os.path.join(base_directory, 'val_' + df_type + '_' + sample_type)
    target_size = (2048, 2048)

    # Set up image data generators
    train_image_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)
    val_image_datagen = ImageDataGenerator(rescale=1./255)
    mask_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

    # Get file lists
    train_file_list = get_file_list(train_hires_folder, train_lowres_folder)
    val_file_list = get_file_list(val_hires_folder, val_lowres_folder)

    # Set up image generators for training set
    train_hiresimage_generator = train_image_datagen.flow_from_dataframe(
            train_file_list,
            x_col='high_res',
            target_size=target_size,
            class_mode = None,
            batch_size = batch_size,
            subset='training',
            shuffle=False)

    train_lowresimage_generator = train_image_datagen.flow_from_dataframe(
            train_file_list,
            x_col='low_res',
            target_size=target_size,
            class_mode = None,
            batch_size = batch_size,
            subset='training',
            shuffle=False)

    # Set up image generators for testing set
    test_hiresimage_generator = train_image_datagen.flow_from_dataframe(
            train_file_list,
            x_col='high_res',
            target_size=target_size,
            class_mode = None,
            batch_size = batch_size,
            subset='validation',
            shuffle=False)

    test_lowresimage_generator = train_image_datagen.flow_from_dataframe(
            train_file_list,
            x_col='low_res',
            target_size=target_size,
            class_mode = None,
            batch_size = batch_size,
            subset='validation',
            shuffle=False)

    # Set up image generators for validation set
    val_hiresimage_generator = val_image_datagen.flow_from_dataframe(
            val_file_list,
            x_col='high_res',
            target_size=target_size,
            class_mode = None,
            batch_size = batch_size,
            shuffle=False)

    val_lowresimage_generator = val_image_datagen.flow_from_dataframe(
            val_file_list,
            x_col='low_res',
            target_size=target_size,
            class_mode = None,
            batch_size = batch_size,
            shuffle=False)

    # Zip image generators for training, validation and testing sets
    train_generator = zip(train_lowresimage_generator, train_hiresimage_generator)
    val_generator = zip(val_lowresimage_generator, val_hiresimage_generator)
    test_generator = zip(test_lowresimage_generator, test_hiresimage_generator)

    # Get sample counts and image generators for training, validation, and testing
    train_samples = train_hiresimage_generator.samples
    val_samples = val_hiresimage_generator.samples
    test_samples = test_hiresimage_generator.samples
    train_img_gen = imageGenerator(train_generator)
    val_img_gen = imageGenerator(val_generator)
    test_img_gen = imageGenerator(test_generator)

    return train_samples, val_samples, test_samples, train_img_gen, val_img_gen, test_img_gen