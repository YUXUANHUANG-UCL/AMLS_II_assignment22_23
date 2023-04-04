import os

def create_data_files():
    file_names = ['train_hr', 'train_x2_bic', 'train_x2_unk', 'train_x3_bic', 'train_x3_unk', 'train_x4_bic', 'train_x4_unk',
                'val_hr', 'val_x2_bic', 'val_x2_unk', 'val_x3_bic', 'val_x3_unk', 'val_x4_bic', 'val_x4_unk']
    for name in file_names:
        if not os.path.exists("Datasets/" + name):
            os.mkdir("Datasets/" + name)
            print("Create data file - " + name)
        
create_data_files()