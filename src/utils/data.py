from pathlib import Path
import os
PATH_TO_FOD2 = Path("datasets/FOD2")

PATH_TO_TRAIN_FILE_LIST = PATH_TO_FOD2 / 'train'/ 'filelist.txt'
PATH_TO_VAL_FILE_LIST = PATH_TO_FOD2 / 'val' / 'filelist.txt'

PATH_TO_FOD2_TRAIN_IMAGES = PATH_TO_FOD2 / 'train' / 'images'
PATH_TO_FOD2_VAL_IMAGES = PATH_TO_FOD2 / 'val' / 'images'

def get_file_list() -> tuple[list[str], list[str]]:
    with open(PATH_TO_TRAIN_FILE_LIST, "r") as file:
        train_file_list = file.readlines()

    with open(PATH_TO_VAL_FILE_LIST, "r") as file:
        val_file_list = file.readlines()

    return train_file_list, val_file_list
    
def update_file_list() -> None:
    filelist_train = []
    filelist_val = []
    for file in os.listdir(PATH_TO_FOD2_TRAIN_IMAGES):
        if file.endswith('.jpg'):
            filelist_train.append(file)
    for file in os.listdir(PATH_TO_FOD2_VAL_IMAGES):
        if file.endswith('.jpg'):
            filelist_val.append(file)

    with open(PATH_TO_TRAIN_FILE_LIST, "w") as file:
        file.write('\n'.join(filelist_train))
    with open(PATH_TO_VAL_FILE_LIST, "w") as file:
        file.write('\n'.join(filelist_val))
    return