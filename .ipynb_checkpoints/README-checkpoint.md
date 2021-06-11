# Assignment 6

## Norm Experiments

The folder contains a runner file called `main.py`. Helper files are present in `./utils/`. Helper files are: 

1. `load_data.py`
    - Contains dataset and sampler functions.
2. `load_model.py`
    - Contains model definition and functions to calculate the number of parameters in the model.
3. `train_loop.py`
    - Train loop for train and validation sets.
4. `test_loop.py`
    - Test loop for test set.



### 1. BatchNorm

#### Stats
Train Accuracy: `99.083%`  
Validation Accuracy: `98.755%`   
Test Accuracy: `99.25%`  


#### Graphs - Loss/Accuracy
!["batchnorm loss/acc"](./../bnorm_graph.png)

#### Misclassified Samples
!["batchnorm mis"](./../bnorm_mis.png)

### 2. Group Norm

#### Stats
Train Accuracy: `98.795%`  
Validation Accuracy: `98.340%`  
Test Accuracy = `98.99%`  

#### Graphs - Loss/Accuracy
!["batchnorm loss/acc"](./../gnorm_graph.png)

#### Misclassified Samples
!["batchnorm mis"](./../gnorm_mis.png)


### 3. Layer Norm

#### Stats
Train Accuracy: `98.147%`  
Validation Accuracy: `97.915%`  
Test Accuracy = `98.27%`  

#### Graphs - Loss/Accuracy
!["batchnorm loss/acc"](./../lnorm_graph.png)

#### Misclassified Samples
!["batchnorm mis"](./../lnorm_mis.png)



### Overall Result

The performance ranking between the three norms are: 

`batch-norm` > `group-norm` > `layer-norm`