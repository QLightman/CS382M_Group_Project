# CS382M_Group_Project
## Clustering Algorithm on Berkeley Dataset and ISBI Dataset
The [Berkeley image segmentation dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html) contains various images that provide different difficulty level for clustering algorithm. The programs in `image_seg_methods` only experiments with selected images.

To run it, make sure correct image file and .mat label file are imported into `img_seg_samples.py`. After that, run
```
python img_seg_samples.py
```

will give the correspond outputs and accuracy values for a specific image.

The `spectral_scale.py` is implemented based on the paper ["A Scalable Spectral Clustering Algorithm Based on Landmark-Embedding and Cosine Similarity"](https://link.springer.com/chapter/10.1007/978-3-319-97785-0_6) by G. Chen (2018). It provides a feasible way to scale spectral clustering into image segmentation.

The [ISBI dataset](http://brainiac2.mit.edu/isbi_challenge/home#:~:text=In%20this%20challenge%2C%20a%20full,and%20small%20image%20alignment%20errors.) consists of 2D segmentation of neuronal processes in EM images. The conventional clustering algorithm has a hard time to perform well on this task, partially because of the fact that intensity-based approach is not enough to extract sufficient features to perform decent segmentation on this dataset. To run this, with correct file path of the .tiff files,
run

```
python ISBI_samples.py
```

will give the encoded .tiff outputs of various clustering algorithm.

## Deep learning based approach

### How to run
You can change the result directory by changing this line
```
saveResult("data/membrane/unet_512_4_EPOCH10",results)
```
You can set different epochs by changing this line
```
model.fit_generator(myGene,steps_per_epoch=300,epochs=10,callbacks=[model_checkpoint])
```
#### Run the Unet model
```
cd deep_learning_based_methods
python Unet.py
```

#### Run the DenseUnet-v1 model
```
cd deep_learning_based_methods
python DenseUnet.py
```

#### Run the DenseUnet-v2 model
```
cd deep_learning_based_methods
python DenseUnet_v2.py
```
### Result for deep learning methods
| Method  | Score |
| ------------- | ------------- |
| Unet  |  [0.938912093](http://brainiac2.mit.edu/isbi_challenge/content/unet5122) |
| denseunet-v1  | [0.964095065](http://brainiac2.mit.edu/isbi_challenge/content/2021422dense2e30)  |
| denseunet-v2  | [waiting](http://brainiac2.mit.edu/isbi_challenge/content/denseunetv2)  |

### Result for conventional clustering methods
| Method  | Score |
| ------------- | ------------- |
| kmeans  |  [0.134271800](http://brainiac2.mit.edu/isbi_challenge/content/kmeans) |
| gaussian_tied  | [0.156796252](http://brainiac2.mit.edu/isbi_challenge/content/gaussiantied)  |
| gaussian_full  | [0.143813315](http://brainiac2.mit.edu/isbi_challenge/content/gaussianfull)  |
| agglo  | [0.141724805](http://brainiac2.mit.edu/isbi_challenge/content/agglo)  |

## Acknowledgements 
We have implemented the deep learning models (Unet, denseunet-v1, denseuent-v2) on our own. And we have utilized the wrapper function inside data.py from this [repo](https://github.com/zhixuhao/unet). It contains some helper function like trainGenerator, testGenerator to help us train the model.
