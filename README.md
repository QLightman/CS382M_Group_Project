# CS382M_Group_Project
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