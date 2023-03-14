# AI-model-to-predict-covid19 using CNN AND Sequential model

#What is the CNN?
The CNN in deep learning means Convolutional Neural Networks. Its is a class of deep neural networks, most commonly applied to analyze visual imagery. Now when we think of a neural network we think about matrix multiplications but that is not the case with ConvNet. It uses a special technique called Convolution. Now in mathematics convolution is a mathematical operation on two functions that produces a third function that expresses how the shape of one is modified by the other.

#Why using CNN instead of normal deep neural network?
Lets say if we have images in size of (1000 X 1000) and there are in RGB. So that means we have 3,000,000 feature, and it is a very big number. Especialy the network will have 3 million input, so lets say the hidden layer will 1,000 unit which leads to the matrix of weights to the first hidden layer is 3,000,000 * 1,000 which will be 3 billion element. This is an enormus number, and needs to alot of very big number of data to avoid Over fiting and need to great computer capabilities.

#what is the image?
To understand how CNN work you first need to understand what is the image and how it works. Any image could be 1D in gray scale case, or it may be 3D in the case of RGB. 1D array may that each pixel in the image will have a singel value. On the contrary, in the case of RGB each pixel will have 3 values: one for red, one for green and one for blue.

Lets talk about grayscale images so it will be more clearier. In this image show what is the convolution is, we have an input image and the kernel is the filter size here it is (3 x 3). Each pixel value is multiplied by its corresponding value, after this All pixel values are summed. This convolution passed to the second convolution and so on..

How does it works?
In the first layer it detect the edges of the image, then in the second layer it detect corners and contours, in the last layer it collect all parts together and it tell you what is this.

MaxPooling
MaxPooling it is the operation that calculates the maximum value in each batch of feature map. It is used for feature reductions. In this example we have (4 x 4) matrix with pool size (2 x 2). So it will split the matrix to 4 mini matrices and will take the biggest value for each batch ,then it will merge them in one final matrix.

DATASETS:Test>>Normal+ Pneumonia
         Training set in 2 different folders-NORMAL+PNEUMONIA 
#RESULTS/PREDICTIONS:

![Screenshot (42)](https://user-images.githubusercontent.com/116704673/225034580-7ef2535b-c5cc-4818-9af4-a428b28a2100.png)
 
 
 The green colour means the person was not sick and model predicted not sick so this correct,the red color means model predicted it wrong.
 
 Code Credit:https://www.kaggle.com/code/hossamrizk/your-first-step-in-cnn (edited in some parts as per requirements)
 Dataset credit:https://www.kaggle.com/datasets/khoongweihao/covid19-xray-dataset-train-test-sets
