import os
import shutil
import pickle

# Define globals
EXCEPT_FOLDER = "_background_noise_"
TRAIN_FOLDER = "train"
VALID_FOLDER = "valid"
TEST_FOLDER = "test"
CLASSES_FILE = "classes.pickle"

def move_files(original_folder, data_folder, data_filename):
    with open(data_filename) as f:
        for line in f.readlines():
            vals = line.split('/')
            dest_folder = os.path.join(data_folder, vals[0])
            if not os.path.exists(dest_folder):
                os.mkdir(dest_folder)
            shutil.move(os.path.join(original_folder, line[:-1]), os.path.join(data_folder, line[:-1]))


def create_train_folder(original_folder, data_folder, test_folder):
    # list dirs
    dir_names = list()
    for file in os.listdir(test_folder):
        if os.path.isdir(os.path.join(test_folder, file)):
            dir_names.append(file)

    # build train folder
    for file in os.listdir(original_folder):
        if os.path.isdir(os.path.join(test_folder, file)) and file in dir_names:
            shutil.move(os.path.join(original_folder, file), os.path.join(data_folder, file))


def make_dataset(gcommands_folder, out_path):
    validation_path = os.path.join(gcommands_folder, 'validation_list.txt')
    test_path = os.path.join(gcommands_folder, 'testing_list.txt')
    directory_contents = os.listdir(gcommands_folder)
    classes = {}
    class_idx = 0
    for _,item in enumerate(directory_contents):
        if os.path.isdir(gcommands_folder+'/'+item):
            if not item == EXCEPT_FOLDER:
              classes[item] = class_idx
              class_idx += 1
    # classes.pop(EXCEPT_FOLDER,None)

    valid_folder = os.path.join(out_path, VALID_FOLDER)
    test_folder = os.path.join(out_path, TEST_FOLDER)
    train_folder = os.path.join(out_path, TRAIN_FOLDER)

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    if not os.path.exists(valid_folder):
        os.mkdir(valid_folder)
    if not os.path.exists(test_folder):
        os.mkdir(test_folder)
    if not os.path.exists(train_folder):
        os.mkdir(train_folder)

    with open(CLASSES_FILE, 'wb') as handle:
        pickle.dump(classes, handle, protocol=pickle.HIGHEST_PROTOCOL)

    move_files(gcommands_folder, test_folder, test_path)
    move_files(gcommands_folder, valid_folder, validation_path)
    create_train_folder(gcommands_folder, train_folder, test_folder)
    shutil.copy(CLASSES_FILE, test_folder)
    shutil.copy(CLASSES_FILE, train_folder)
    shutil.copy(CLASSES_FILE, valid_folder)