from PHC import PHC, preprocess
import numpy as np
from tqdm import tqdm

#Generate various PHC datasets with custom parameters
classes = ["alpha", "lower_star", "ext_adj_complex"]
thresh_param = [195, 200]
dimension = [0,1]

histo_imgs = np.load("grey_ost_img/balanced_img.npy", allow_pickle=True)
lb = np.load("grey_ost_img/balanced_lb.npy", allow_pickle=True)
for type in classes:
    for param in thresh_param:
        for dim in dimension:
            data_type = "Computing PHC - "+type+" t"+str(param)+" dim "+str(dim)
            conditioning = preprocess(thresh=param, iterate=1)
            PHC_data = np.empty((1143,20,20)) 
            for img in tqdm(range(len(histo_imgs)), mininterval=5.0, desc=data_type):
                thresh_img = conditioning.threshold(histo_imgs[img])
                prepped_image = conditioning.dilate(thresh_img) 
                localhom = PHC(persistence_type=type, stride=512, window_size=512, dimension=dim) 
                data = localhom.convolve(prepped_image)
                data = np.array(data).reshape((20,20))
                PHC_data[img] = data
            
            print("Saving File")
            np.save(str(dim)+"_"+type+"_PHC_data_t"+str(param), PHC_data, allow_pickle=True)







