# Classification of Histopathology Slides with Persistent Homology Convolutions
GitHub respository for those interested in generating data that captures localized peristence of biological micrographs or any general image data.
Check out the corresponding paper at: https://arxiv.org/abs/2507.14378

![alt text](https://github.com/Shrunalp/PHC/blob/main/PHC_visual.png?raw=true#center)

## Abstract
Convolutional neural networks (CNNs) are a standard tool for computer vision tasks such as image classification. However, typical model architectures may result in the loss of topological information. In specific domains such as histopathology, topology is an important descriptor that can be used to distinguish between disease-indicating tissue by analyzing the shape characteristics of cells. Current literature suggests that reintroducing topological information using persistent homology can improve medical diagnostics; however, previous methods utilize global topological summaries which do not contain information about the locality of topological features. To address this gap, we present a novel method that generates local persistent homology-based data using a modified version of the convolution operator called Persistent Homology Convolutions. This method captures information about the locality and translation invariance of topological features. We perform a comparative study using various representations of histopathology slides and find that models trained with persistent homology convolutions outperform conventionally trained models and are less sensitive to hyperparameters. These results indicate that persistent homology convolutions extract meaningful geometric information from the histopathology slides.

## Installation

Clone the Github repoistory:
```
git clone https://github.com/Shrunalp/PHC.git
```
or using Git CLI 
```
gh repo clone Shrunalp/PHC
```


## Tutorial 

Check out our notebook ```PHC_tutorial.ipynb``` to get started!

## Authors

Shrunal Pothagoni - spothago@gmu.edu

Benjamin Schweinhart - bschwei@gmu.edu
