import numpy as np
from local_homology import gCNN
from tqdm import tqdm
import time
import os

def main():
    Data = np.load("grey_ost_img/balanced_img.npy")

    file_name = "lowerstar_new"
    persist = gCNN(cmplx="sublevel", vector="img", stride=32, patch=32, res=20, fltr=200, itr=1, kernel=2)
    np.save(file_name, np.array([persist.conv_pers(Data[j]) for j in tqdm(range(len(Data)), desc="Processing images")], dtype=object))


if __name__=="__main__":
    main()