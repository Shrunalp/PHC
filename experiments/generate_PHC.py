from PHC import PHC, preprocess
import numpy as np
from tqdm import tqdm

histo_imgs = np.load("grey_ost_img/balanced_img.npy", allow_pickle=True)
lb = np.load("grey_ost_img/balanced_lb.npy", allow_pickle=True)
conditioning = preprocess(thresh=200, iterate=1)
PHC_data = np.empty((1143,256,20,20)) 

for img in tqdm(range(len(histo_imgs)), desc="Computing PHC"):
    thresh_img = conditioning.threshold(histo_imgs[img])
    prepped_image = conditioning.dilate(thresh_img) 
    localhom = PHC(persistence_type="extended", stride=32, window_size=32) 
    data = localhom.convolve(prepped_image)
    data = np.array(data).reshape((256,20,20))
    PHC_data[img] = data

print("Saving File")
np.save("extended_PHC_data", PHC_data, allow_pickle=True)



