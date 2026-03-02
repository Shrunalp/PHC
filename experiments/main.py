import skdim
import numpy as np
import torch

MLE = skdim.id.MLE()
data = ["alpha_global"]

for data_type in data:
    PHC_data = np.array(np.load(data_type+".npy", allow_pickle=True)[:500], dtype=np.float64).reshape(500, 20, 20)
    np.shape(PHC_data)
    dimensions = []
    for i in range(500):
        dim = MLE.fit_predict(PHC_data[i])
        if torch.isnan(torch.tensor(dim)):
            continue
        elif dim==0.0:
            continue
        else:
            dimensions.append(dim)

    print("MLE ID - "+data_type+":", np.mean(dimensions))