import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, add

def psnr(y_true, y_pred):
    """Calculates the Peak Signal to Noise Ratio (PSNR) between two images.

    Args:
        y_true: The ground truth image.
        y_pred: The predicted image.

    Returns:
        The PSNR value between the two images.
    """
    # Set the maximum pixel value to 1, since the pixel values are assumed to be in the range [0, 1].
    max_pixel = 1
    # Calculate the Mean Squared Error (MSE) between y_true and y_pred.
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    # Calculate the PSNR value using the MSE.
    psnr = 20 * tf.math.log(max_pixel / tf.math.sqrt(mse)) / tf.math.log(10.0)
    return psnr

def train_autoencoder_hr(train_samples, val_samples, train_img_gen, val_img_gen, model_path, fig_path, batch_size):
    """Trains an autoencoder model on super-resolution tasks.

    Args:
        train_samples (int): The total number of training samples.
        val_samples (int): The total number of validation samples.
        train_img_gen (generator): The training image generator.
        val_img_gen (generator): The validation image generator.
        model_path (str): The file path to save the trained model.
        fig_path (str): The file path to save the training plot.
        batch_size (int): The batch size for training.

    Returns:
        The trained autoencoder model.
    """
    # Define the model architecture
    input_img = Input(shape=(2048, 2048, 3))

    # Encoder layers
    l1 = Conv2D(64, (3, 3), padding='same', activation='relu')(input_img)
    l2 = Conv2D(64, (3, 3), padding='same', activation='relu')(l1)
    l3 = MaxPooling2D(padding='same')(l2)
    l3 = Dropout(0.3)(l3)
    l4 = Conv2D(128, (3, 3),  padding='same', activation='relu')(l3)
    l5 = Conv2D(128, (3, 3), padding='same', activation='relu')(l4)
    l6 = MaxPooling2D(padding='same')(l5)

    # Bottleneck layer
    l7 = Conv2D(256, (3, 3), padding='same', activation='relu')(l6)

    # Decoder layers
    l8 = UpSampling2D()(l7)
    l9 = Conv2D(128, (3, 3), padding='same', activation='relu')(l8)
    l10 = Conv2D(128, (3, 3), padding='same', activation='relu')(l9)
    l11 = add([l5, l10])
    l12 = UpSampling2D()(l11)
    l13 = Conv2D(64, (3, 3), padding='same', activation='relu')(l12)
    l14 = Conv2D(64, (3, 3), padding='same', activation='relu')(l13)
    l15 = add([l14, l2])

    decoded = Conv2D(3, (3, 3), padding='same', activation='relu')(l15)

    autoencoder = Model(input_img, decoded)
    
    # Compile the model
    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=[psnr])

    autoencoder.summary()

    # Define the model callbacks
    checkpoint = ModelCheckpoint(model_path,
                                monitor="val_loss",
                                mode="min",
                                save_best_only = True,
                                verbose=1)

    earlystop = EarlyStopping(monitor = 'val_loss', 
                                min_delta = 0, 
                                patience = 5,
                                verbose = 1,
                                restore_best_weights = True)

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                                patience=3, 
                                                verbose=1, 
                                                factor=0.2, 
                                                min_lr=0.00000001)
    
    # train
    hist = autoencoder.fit(train_img_gen,
                        steps_per_epoch=train_samples//batch_size,
                        validation_data=val_img_gen,
                        validation_steps=val_samples//batch_size,
                        epochs=10, callbacks=[earlystop, checkpoint, learning_rate_reduction])
    
    # plot hist
    psnr_hist = hist.history['psnr']
    val_psnr = hist.history['val_psnr']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    epochs_range = range(1, len(hist.epoch) + 1)
    plt.figure(figsize=(15,5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, psnr_hist, label='Train Set')
    plt.plot(epochs_range, val_psnr, label='Validation Set')
    plt.legend(loc="best")
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.title('Model PSNR')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Set')
    plt.plot(epochs_range, val_loss, label='Validation Set')
    plt.legend(loc="best")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.savefig(fig_path)

    return autoencoder