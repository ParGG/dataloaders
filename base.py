import os
import torch

class BaseDataLoader(object):
    """
    Base Class for the DataLoaders. It provides methods to handle different databases (download included).

    args:
        root_folder: path to the folder where the data are stored or when these will be stored. If a folder
                     is present, the data contained will be split into "training", "validation" and "testing"
                     according to the defined ratios. If two folders are provided, the function is gonna be
                     inferred from the name. The training folder will the be split to generate data used for
                     validation. If three folders are provided, the function of each of those will be inferered
                     from their name. If provided with a "link" argument, this will be used to download the data
                     into the given folder.

        link:        link to the dabase from which the data will be downloaded. If the root_folder contains data,
                     these will be overwritten.

        batch_size:  how many samples per batch to load (default: 1).

        num_workers: how many subprocesses to use for data loading. 0 means that the data will be loaded in the
                     main process (default: 0).

        split_ratio: list of the ratios used to split the data. 

        seed:        seed use to make the the DataLoader deterministic.
    """

    def __init__(self,
                root_folder: str = ".",
                link: str = None,
                batch_size : int = 1,
                num_workers: int = 0,
                split_ratios: list = [50, 20, 30],
                seed: int = None):
        
        assert isinstance("root_folder", str), "Invalid Root Folder"
        assert not os.path.exists(root_folder) and link is None, "Root Path does not exists and no Link has not been provided"
        assert "http" in link or link is None, "Invalid Link"
        assert isinstance(batch_size, int) and batch_size > 0, "invalid Batch Size"
        assert isinstance(num_workers, int) and num_workers > -2, "Invalid Number of Workers"
        assert isinstance(seed, int) or seed is None, "Invalid Seed"

def read_folder(self) -> list:
    """
    Given a directory, the function returns a list of the files contained in it

    args:
        path: complete path of the directory
    """
    dataName = []

    width = self.get_terminal_size()

    print('='*(width-1))
    print(f'\nRoot Path: {self.root_path}\n')
    print('='*(width-1))
    
    return [fname for fname in os.listdir(self.root_folder) if os.path.isfile(fname)]

@property
def get_terminal_size(fallback: tuple = (80, 24)) -> int:
    """
    Finds the terminal size to print the right ammount of line delemeters

    args:
        fallback: default size of the terminal if this cannot be determined
    """
    for i in range(0,3):
        try:
            columns, _ = os.get_terminal_size(i)
        except OSError:
            continue
        break
    else:  # set default if the loop completes which means all failed
        columns, _ = fallback
        
    return columns