import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow import keras
import matplotlib.patches as patches
import mpl_toolkits.axes_grid1.inset_locator as mpl_il

def ssimiq(img1, img2):
    """Computes the Structural Similarity Index (SSIM) between two grayscale images.
    
    Args:
        img1: A numpy array representing the first grayscale image.
        img2: A numpy array representing the second grayscale image.
        
    Returns:
        A float representing the SSIM score between the two images.
    """
    # Convert images to grayscale
    img1 = np.dot(img1[...,:3], [0.299, 0.587, 0.114])
    img2 = np.dot(img2[...,:3], [0.299, 0.587, 0.114])
    
    # Constants for SSIM calculation
    K1 = 0.001
    K2 = 0.003
    L = 1.0
    
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    
    # Compute means and variances
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.cov(img1, img2)[0][1]
    
    # Compute SSIM
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    
    # Ensure the SSIM value is between -1 and 1
    ssim_score = np.clip(numerator / denominator, -1, 1)
    
    return ssim_score

def ssimiq_datasets(img_set1, img_set2):
    """Computes the average Structural Similarity Index (SSIM) between two sets of grayscale images.
    
    Args:
        img_set1 (np.ndarray): Set of grayscale images to compare. Shape: (num_images, height, width).
        img_set2 (np.ndarray): Set of grayscale images to compare. Shape: (num_images, height, width).
        
    Returns:
        float: Average SSIM between the two image sets.
    """
    # Convert image sets to grayscale
    img_set1_gray = np.dot(img_set1[...,:3], [0.299, 0.587, 0.114])
    img_set2_gray = np.dot(img_set2[...,:3], [0.299, 0.587, 0.114])
    
    # Constants for SSIM calculation
    K1 = 0.001
    K2 = 0.003
    L = 1.0
    
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    
    # Compute means and variances
    mu1 = np.mean(img_set1_gray, axis=(1, 2))
    mu2 = np.mean(img_set2_gray, axis=(1, 2))
    sigma1_sq = np.var(img_set1_gray, axis=(1, 2))
    sigma2_sq = np.var(img_set2_gray, axis=(1, 2))
    sigma12 = np.array([np.cov(img_set1_gray[i,:,:].flatten(), img_set2_gray[i,:,:].flatten())[0][1] for i in range(len(img_set1_gray))])
    
    # Compute SSIM
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    
    # Ensure the SSIM values are between -1 and 1
    ssim_scores = np.clip(numerator / denominator, -1, 1)
    
    # Return average SSIM
    return np.mean(ssim_scores)


def predict_img(test_generator, model_path, psnr, fig_path, name):
    """
    Predicts high-resolution images from low-resolution images using a pre-trained model.

    Args:
        test_generator: The generator object that produces the low-resolution test images.
        model_path: The path to the saved pre-trained model.
        psnr: The function that calculates the PSNR value between two images.
        fig_path: The path to the directory where the predicted images will be saved.
        name: The name to use for the saved images.

    Returns:
        None
    """
    # Load the pre-trained model
    model = keras.models.load_model(model_path, custom_objects = {'psnr': psnr})
    # Get the first batch of images in the test_generator
    x_test, y_test = next(test_generator, 1)
    # Predict the high-res images from the low-res images
    y_pred = model.predict(x_test)
    # Define the coordinates and size of the rectangle
    x, y = 900, 900
    width, height = 300, 300
    print('Average Evaluation Values for y_pred:')
    # Calculate the PSNR values
    psnr_values = psnr(y_test, y_pred)
    print('PSNR: %f'%psnr_values)
    mse = np.mean((y_test - y_pred)**2)
    print('MSE: %f'%mse)
    ssim_datasets = ssimiq_datasets(y_test, y_pred)
    print('SSIM: %f'%ssim_datasets)
    
    # For x_test
    print('Average Evaluation Values for x_test:')
    psnr_values = psnr(y_test, x_test)
    print('PSNR: %f'%psnr_values)
    mse = np.mean((y_test - x_test)**2)
    print('MSE: %f'%mse)
    ssim_datasets = ssimiq_datasets(y_test, x_test)
    print('SSIM: %f'%ssim_datasets)
    
    # Loop through the 2nd and 4th images and plot them along with their corresponding images and PSNR values
    for i in [2,4]:
        fig, axs = plt.subplots(1, 3, figsize=(20, 10))
        
        # Plot the high-res image
        axs[0].imshow(y_test[i])
        axs[0].set_title('High-Res Image')
        
        # Plot the predicted image
        psnr_pred = psnr(y_test[i], y_pred[i])
        ssim_pred = ssimiq(y_test[i], y_pred[i])
        axs[1].imshow(y_pred[i])
        axs[1].set_title('Predicted Image PSNR/SSIMIQ - ' + str(psnr_pred.numpy()) + r'/' + str(round(ssim_pred,7)))
        
        # Plot the low-res image
        psnr_lr = psnr(y_test[i], x_test[i])
        ssim_lr = ssimiq(y_test[i], x_test[i])
        axs[2].imshow(x_test[i])
        axs[2].set_title('Low-Res Image PSNR/SSIMIQ - ' + str(psnr_lr.numpy()) + r'/' + str(round(ssim_lr,7)))
        
        rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
        for ax in axs:
            ax.add_patch(patches.Rectangle(rect.get_xy(), rect.get_width(), rect.get_height(), linewidth=1, edgecolor='r', facecolor='none'))
            ax.relim() # Update the limits of the axes
            ax.autoscale() # Auto-scale the view limits to the data
        
        # Add inset to show enlarged view of region inside rectangle
        axins = mpl_il.inset_axes(axs[0], width="40%", height="40%", loc='lower right')
        axins.imshow(y_test[i][y:y+height, x:x+width])
        axins.set_xticks([])
        axins.set_yticks([])
        
        axins = mpl_il.inset_axes(axs[1], width="40%", height="40%", loc='lower right')
        axins.imshow(y_pred[i][y:y+height, x:x+width])
        axins.set_xticks([])
        axins.set_yticks([])
        
        axins = mpl_il.inset_axes(axs[2], width="40%", height="40%", loc='lower right')
        axins.imshow(y_test[i][y:y+height, x:x+width])
        axins.set_xticks([])
        axins.set_yticks([])
        
        plt.savefig(os.path.join(fig_path, str(i) + '_' + name + '.jpg'))