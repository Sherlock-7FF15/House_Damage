# House_Damage
Introduction to Neural Network Online Project-Based Learning Program

# Motivation about the Project

The latest hurricane - Hurricane Iota, had 61 total fatalities, and 41 are still missing. After a hurricane, damage assessment is vital to the relief helpers and first responders so that resources and help can be planned and allocated appropriately. One way to measure the damage is to detect and quantify the number of damaged buildings, usually done by driving around the affected area and noting down manually. This process can be labor-intensive and time-consuming and not the most efficient method as well. In that case, in the following project, we will find an efficient convolution neural network to identify whether a house was damaged using the satellite imagery as well as the best way to improve it’s appearance.

# Data and Labels

Our data set is constructed of 16,000 RGB satellite imagery before or after the house was damaged. Before we use these images to train the model, we normalized these images in order to facilitate model prediction.

![img](file:///C:/Users/Soviet/AppData/Local/Temp/msohtmlclip1/01/clip_image002.jpg)

*Dataset sample*

These satellite imagery were divided into the following sub data set.

ü  train_another: the training data; 5000 images of each class(damage/no damage)

ü  validation_another: the validation data; 1000 images of each class(damage/no damage)

ü  test_another: the unbalanced test data; 8000/1000 images of damaged/undamaged classes

ü  test: the balanced test data; 1000 images of each class(damage/no damage)

 

# Model

After comparing the VGG16, AlexNet and many other models as well as their simplified forms, we finally choose the model structure and hyper-parameters as follows:

![img](file:///C:/Users/Soviet/AppData/Local/Temp/msohtmlclip1/01/clip_image004.jpg)

*Model Structure*

 

![img](file:///C:/Users/Soviet/AppData/Local/Temp/msohtmlclip1/01/clip_image006.jpg)

*Hyper-parameters*

 

![图片1](file:///C:/Users/Soviet/AppData/Local/Temp/msohtmlclip1/01/clip_image008.jpg)

*VGG16 Structure*

As can be seen from the structure in the figures above, our model combines the convolution layer of VGG16 pre-training model and our self-defined three Dense layers. When we fit the model, the convolution layer of the model is set as untrainable in order not to affect the pre-training effect.

 

## Bootstrapping

Bootstrapping is any test or metric that uses random sampling with replacement and falls under the broader class of resampling methods. The process of Bootstrapping is as followed:

![img](file:///C:/Users/Soviet/AppData/Local/Temp/msohtmlclip1/01/clip_image010.jpg)

*Bootstrapping*

To apply bootstrap, we firstly load the original dataset with some data preprocessing. Here we use normalization to limit the data form 0 to 1. Bootstrap then extracts many subdatasets from the original data with put back. Each subdataset will be trained by our CNN model.

Bootstrap fully makes use of original data and increases the amount of datasets, which is effective for small dataset and is helpful for further ensemble learning. Then my partner Jiahe will further explaining bagging.

 

## Bagging

Bootstrap aggregating, also called bagging (from bootstrap aggregating), is a [machine learning ensemble](https://en.wikipedia.org/wiki/Ensemble_learning) [meta-algorithm](https://en.wikipedia.org/wiki/Meta-algorithm) designed to improve the [stability](https://en.wikipedia.org/wiki/Stability_(learning_theory)) and accuracy of [machine learning](https://en.wikipedia.org/wiki/Machine_learning) algorithms used in [statistical classification](https://en.wikipedia.org/wiki/Statistical_classification) and [regression](https://en.wikipedia.org/wiki/Regression_analysis). It also reduces [variance](https://en.wikipedia.org/wiki/Variance) and helps to avoid [overfitting](https://en.wikipedia.org/wiki/Overfitting). Although it is usually applied to [decision tree](https://en.wikipedia.org/wiki/Decision_tree_learning) methods, it can be used with any type of method. Bagging is a special case of the [model averaging](https://en.wikipedia.org/wiki/Ensemble_learning) approach.

The process of bagging is as followed:

![img](file:///C:/Users/Soviet/AppData/Local/Temp/msohtmlclip1/01/clip_image012.jpg)

*Bagging*

Firstly, the data is loaded from the original dataset and preprocessed, then it is shuffled to generate multiple subdatasets of the same size as the original dataset. Secondly, a new basic model is trained and saved for each subdataset. Thirdly, these well trained basic models are used to predict the test set, and the same number of prediction data as the basic models can be obtained on a single data. Finally, we used a simple voting strategy to determine the final prediction.

 

## Occlusion

We apply the occlusion methodology to visualize the performance of the VGG_16 model. Occlusion map is a forward’s method which tells the activation level of each pixel of the input images, showing which part of the input our trained model is interested at. In the occlusion map, we set the grey patch of size to be 30px*30px. The grey patch will start at the origin of the image (0px,0px) and move 5px each iteration to the left or below. For each iteration, a loss map will be generated, which will be summed up together to create the occlusion map.

 

 

# Result

## Model Evaluation Result

|                | **VGG_16(pre)** | **VGG_16**   **(****Bootstrapping)** | **VGG_16**   **(Bagging)** | **Alex_Net** |
| -------------- | --------------- | ------------------------------------ | -------------------------- | ------------ |
| Unbal_accuracy | 0.9789          | 0.9487                               | 0.9511                     | 0.9319       |
| Accuracy       | 0.9410          | 0.9635                               | 0.9735                     | 0.9460       |
| Ubal_precision | 0.9865          | 0.9636                               | 0.9513                     | 0.9952       |
| Precision      | 0.9019          | 0.9478                               | 0.9660                     | 0.9515       |
| Ubal_recall    | 0.9897          | 0.9893                               | 0.9933                     | 0.9274       |
| Recall         | 0.9884          | 0.9732                               | 0.9807                     | 0.9387       |
| Ubal_F1        | 0.9879          | 0.9754                               | 0.9718                     | 0.9594       |
| F1             | 0.9408          | 0.9643                               | 0.9732                     | 0.9434       |

*Model Efficiency*

## ![img](file:///C:/Users/Soviet/AppData/Local/Temp/msohtmlclip1/01/clip_image014.jpg)

*Comparison of Model Efficiency*

Four models designed in the previous section are implemented, including VGG_16,  VGG_16 with Bootstrapping, VGG_16 with bagging and AlexNet. The models are then tested both on the balanced testing dataset and the unbalanced testing dataset. In the testing, accuracy, precision, recall and f1 score are evaluated. The results are aggregated and reflected in the table above.  

 

It is significant that all accuracies of the four models we implemented are higher than 0.9 both on balanced and unbalanced testing dataset. The VGG_16 performs slightly worse than AlexNet on balanced dataset but much better on unbalanced testing dataset. However, it is hard to decide which model performs better only based on the result listed above.

 

Bootstrapping and Bagging method have slight improvement on both the accuracy, precision, recall and f1 over the original VGG_16 model based on the balanced model. However, the performance gets worse when testing on the unbalanced data.

 

## Occlusion Result

![Graphical user interface  Description automatically generated](file:///C:/Users/Soviet/AppData/Local/Temp/msohtmlclip1/01/clip_image016.jpg) ![Graphical user interface, application  Description automatically generated](file:///C:/Users/Soviet/AppData/Local/Temp/msohtmlclip1/01/clip_image018.jpg)

​       *Occlusion Map of Undamaged building         Occlusion Map of damaged building*

It is interesting to notice that VGG_16 tent to focus on the buildings or the trees, which have distinguishable border from the environment. We can conclude that VGG_16 model learns important features of the images based on the fact. On the contrary, the Alex Net is focus on the unimportant information about the image. Therefore, we can indicate that the VGG_16 may perform better than Alex Net. 

#  

#  

# Conclusion and Future Work

In conclusion, in this article, we design, enhance, improve and visualize the VGG_16 Convolution Neural Network, which classifies damaged and undamaged buildings based on the satellite images post hurricane. The VGG_16 model has a good performance with classification accuracy of 94.1%. The performance is further enhanced by Bootstrapping and bagging. During the visualization, the VGG_16 can extract the significant objects from the input images and is proved to be “learning”. 

 

In the future, Bootstrapping and bagging method will be further adjusted to construct a firm and robust model which can improve the accuracy approaching to 100%. Other preprocessing method and normalization methods will also be applied to enhance the performance. Meanwhile, location features can be added to help with the classification. 