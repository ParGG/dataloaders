import torchaudio
import torch as th

import os
import shutil
import pickle
import argparse

from typing import Tuple
from nnchassis import PrintUtils as P

from torch import utils
from torchaudio import transforms
from torchaudio.datasets.utils import walk_files

from .dlutils import make_dataset

# Define globals
HASH_DIVIDER = "_nohash_"
EXCEPT_FOLDER = "_background_noise_"
TRAIN_FOLDER = "train"
VALID_FOLDER = "valid"
TEST_FOLDER = "test"
CLASSES_FILE = "classes.pickle"
SAMPLE_LENGTH = 16000

class GoogleCommandsDataset(utils.data.Dataset):
    """
    Create a Dataset for Speech Commands. Each item is a tuple of the form:
    waveform, sample_rate, label, speaker_id, utterance_number

    args:
        root_path: path to the folder where the data are located
    """

    def __init__(self,
                 root_path: str) -> None:

        root_path.replace("\\", "/")

        self._path = root_path

        classes_file_path = os.path.join(root_path, CLASSES_FILE)

        walker = walk_files(self._path, suffix= ".wav", prefix= True)
        walker = filter(lambda w: HASH_DIVIDER in w and EXCEPT_FOLDER not in w, walker)
        self._walker = list(walker)

        with open(classes_file_path, 'rb') as handle:
            self.classes = pickle.load(handle)

    def __getitem__(self, n: int) -> Tuple[th.Tensor, int, str, str, int]:

        fileid = self._walker[n]
        return self.load_speechcommands_item(fileid, self._path)

    def __len__(self) -> int:

        return len(self._walker)

    def load_speechcommands_item(self,
                                filepath: str,
                                path: str) -> Tuple[th.Tensor, int, str, str, int]:
        
        relpath = os.path.relpath(filepath, path)
        label, filename = os.path.split(relpath)
        speaker, _ = os.path.splitext(filename)

        speaker_id, utterance_number = speaker.split(HASH_DIVIDER)
        utterance_number = int(utterance_number)

        # Load audio
        waveform, sample_rate = torchaudio.load(filepath)

        if (waveform.shape[1] < SAMPLE_LENGTH):
            # pad early with zeros in case the sequence is shorter than 16000 samples
            wave = th.zeros([1,SAMPLE_LENGTH])
            wave[0, -waveform.shape[1]:] = waveform
            waveform = wave

        waveform = th.squeeze(waveform)

        # return waveform, sample_rate, label, speaker_id, utterance_number
        return waveform, self.classes[label]    

def gen_dataloaders(root_path: str,
                    train_batch_size: int = 32, 
                    test_batch_size: int = 32, 
                    val_batch_size: int = 32,
                    download: bool = False) -> utils.data.DataLoader:
    """
    Generates a DataLoader for the Google Speech Commands task. Each item is a tuple of the form:
    (waveform, label).

    args:
        root_path: path to the folder where the data are stored
        train_batch_size: batch size used for the dataloader for training
        test_batch_size: batch size used for the dataloader for testing
        val_batch_size: batch size used for the dataloader for validation
        download: if True, downloads the dataset (default: True)

    
    return:
        tuple(train_dataloader, validation_dataloader, test_dataloaders)
    """

    root_path = root_path.replace("\\", "/")

    src_path = os.path.join(root_path, 'SpeechCommands', 'speech_commands_v0.02')
    out_path = os.path.join(root_path, "SpeechCommands")

    if not download:
      if not os.path.exists(out_path):
        P.print_message("Looks like dataset folder is missing. You may want check and enable download")
    else:
      # first download the dataset
      download_marker_file = os.path.join(root_path, "downloaded")

      if os.path.isfile(download_marker_file):
        x = input("Database seems downloaded. Wipe and download [y|n]?:")
        download = True if (x == "y" or x == "Y") else False

      if download:
          shutil.rmtree(root_path)
          os.mkdir(root_path)

          # Download the dataset using the default dataset from Pytorch audio
          _ = torchaudio.datasets.SPEECHCOMMANDS(root= root_path, download= True)

          # then separate the test, train and validation sets
          make_dataset(gcommands_folder= src_path, out_path= out_path)
          dld_mark = open(download_marker_file, "w")
          dld_mark.write("True")
          dld_mark.close()

    # Now create thed dataloaders
    train_path = os.path.join(out_path, TRAIN_FOLDER)
    train_dataset = GoogleCommandsDataset(root_path= train_path)

    val_path = os.path.join(out_path, VALID_FOLDER)
    val_dataset = GoogleCommandsDataset(root_path= val_path)

    test_path = os.path.join(out_path, TEST_FOLDER)
    test_dataset = GoogleCommandsDataset(root_path= test_path)

    train_dataloader = utils.data.DataLoader(train_dataset, 
                                            batch_size= train_batch_size,
                                            shuffle= True)
    
    val_dataloader = utils.data.DataLoader(val_dataset, 
                                            batch_size= val_batch_size, 
                                            shuffle= True)

    test_dataloader = utils.data.DataLoader(test_dataset, 
                                            batch_size= test_batch_size, 
                                            shuffle= False)

    return train_dataloader, val_dataloader, test_dataloader

def gscommands_gen(root_path= None, 
                    train_batch_size: int = 32, 
                    test_batch_size: int = 32, 
                    val_batch_size: int = 32,
                    download: bool = False):
    """
    Generates DataLoaders for the Google Speech Commands task. Each item is a tuple of the form:
    (waveform, label).

    args:
        root_path: path to the folder where the data are stored
        train_batch_size: batch size used for the dataloader for training
        test_batch_size: batch size used for the dataloader for testing
        val_batch_size: batch size used for the dataloader for validation
        download: if True, downloads the dataset (default: True)

    return:
        tuple(train_dataloader, validation_dataloader, test_dataloaders)
    """

    if root_path is None:
        root_path = os.path.join(os.getcwd(), "dataset")
        if not os.path.exists(root_path):
            os.mkdir(root_path)

    return gen_dataloaders(root_path= root_path, 
                        download= download, 
                        train_batch_size= train_batch_size,
                        test_batch_size= test_batch_size,
                        val_batch_size= val_batch_size)
