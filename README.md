# ELEC0135 Applied Machine Learning Systems II (22/23) Assignment

## How to run the code for image super-resolution?
### All pretrained models are preserved in corresponding folders (X2, X3, X4 and lr_model)!!!

1. cd to the AMLS_II_assignment22_23 path:
```
cd your/file/path/AMLS_II_assignment22_23
```

2. create the necessary environment:
```
conda env create -f environment.yml
```

3. activate conda environment
```
conda activate amls2
```

4. create datasets file for training and then you need to move datasets to corresponding folders (see detailed information the README.md in folder **/Datasets**:

```
python creat_datasets_file.py
```

5. start the code:

```
# if you don't have pretrained model in X2, X3 and X4 folder, you can run:
python main.py
```

```
# or you can directly run:
python pred.py
```

6. start the super-resolution task for ultra-low-resolution images:
```
python pred_img_lr.py
```

## Role of each folder and file

**Datasets** contains all datasets and they are stored in a structure.

**lr_model** contains pretrained model for the super-resolution task of ultra-low-resolution images and its result.

**X2** contains pretrained model for the super-resolution task of x2 downsampled images and its result.

**X3** contains pretrained model for the super-resolution task of x3 downsampled images and its result.

**X4** contains pretrained model for the super-resolution task of x4 downsampled images and its result.

**create_datasets_file.py** includes functions to create a folder structure for datasets.

**data_generator.py** includes functions to import images from datasets and will be input in the model.

**autoencoder_hr.py** includes functions to train an autoencoder model for image super-resolution using x2, x3 and x4 downsampled images.

**predict_img.py** includes functions to load models to predict high-resolution images from low-resolution images and visulise results.

**main.py** is the file to start the whole process of the assignment but not includes the ultra-low-resolution image super-resolution task.

**pred_img_lr.py** is the file to train the model for super-resolution of ultra-low-resolution images to prove that the model can finish the super-resolution task by smoothing mosaic pixels and has a good performance dealing the such images.










