from data_generator import train_val_generator
from autoencoder_hr import psnr
from predict_img import predict_img
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf

###########################################
# This python file is created only for model prediction if all .h5 weight files exists.
###########################################

# ======================================================================================================================
batch_size = 5
base_path = ''
# ======================================================================================================================
# x2 bic
train_samples, val_samples, test_samples, train_img_gen, val_img_gen, test_img_gen = train_val_generator('x2', 'bic', batch_size)
model_path = os.path.join(base_path, 'X2/x2_bic_model.h5')
pred_path = os.path.join(base_path, 'X2')
pred = predict_img(test_img_gen, model_path, psnr, pred_path, 'bic')

# ======================================================================================================================
# x2 unk
train_samples, val_samples, test_samples, train_img_gen, val_img_gen, test_img_gen = train_val_generator('x2', 'unk', batch_size)
model_path = os.path.join(base_path, 'X2/x2_unk_model.h5')
pred_path = os.path.join(base_path, 'X2')
pred = predict_img(test_img_gen, model_path, psnr, pred_path, 'unk')

# ======================================================================================================================
# x3 bic
train_samples, val_samples, test_samples, train_img_gen, val_img_gen, test_img_gen = train_val_generator('x3', 'bic', batch_size)
model_path = os.path.join(base_path, 'X3/x3_bic_model.h5')
pred_path = os.path.join(base_path, 'X3')
pred = predict_img(test_img_gen, model_path, psnr, pred_path, 'bic')
# ======================================================================================================================
# x3 unk
train_samples, val_samples, test_samples, train_img_gen, val_img_gen, test_img_gen = train_val_generator('x3', 'unk', batch_size)
model_path = os.path.join(base_path, 'X3/x3_unk_model.h5')
pred_path = os.path.join(base_path, 'X3')
pred = predict_img(test_img_gen, model_path, psnr, pred_path, 'unk')
# ======================================================================================================================
# x4 bic
train_samples, val_samples, test_samples, train_img_gen, val_img_gen, test_img_gen = train_val_generator('x4', 'bic', batch_size)
model_path = os.path.join(base_path, 'X4/x4_bic_model.h5')
pred_path = os.path.join(base_path, 'X4')
pred = predict_img(test_img_gen, model_path, psnr, pred_path, 'bic')

# ======================================================================================================================
# x4 unk
train_samples, val_samples, test_samples, train_img_gen, val_img_gen, test_img_gen = train_val_generator('x4', 'unk', batch_size)
model_path = os.path.join(base_path, 'X4/x4_unk_model.h5')
pred_path = os.path.join(base_path, 'X4')
pred = predict_img(test_img_gen, model_path, psnr, pred_path, 'unk')
