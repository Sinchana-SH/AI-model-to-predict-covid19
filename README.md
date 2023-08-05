## AI-model-to-predict-covid19 using Chest Xrays using CNN,Sequential model and comparison with other pretrained models.

#Objective
The project 's motive is to create a comprehensive analysis detection methodology based on Deep learning for COVID-19 diagnosis. The objectives of this project are summarized in the following points,
• Study and comprehend the noteworthy Deep learning methods for COVID-19 diagnosis.
• To establish the financial Overhead for detecting COVID-19 using traditional methods and improving the detection method using deep learning techniques.


Technologies used:
>Artificial intelligence and Deep learning : Artificial Intelligence (AI) is the study and creation of computer systems that can perceive, reason and act. The primary aim of Al is to produce intelligent machines. The intelligence should be exhibited by thinking, making decisions, solving problems, more importantly by learning. Al is an interdisciplinary field that requires knowledge in computer science, linguistics, psychology, biology, philosophy and so on for serious
 

>Deep learning is an artificial intelligence (AI) function that imitates the workings of the human brain in processing data and creating patterns for use in decision making. Deep learning is a subset of machine learning in artificial intelligence that has networks capable of learning unsupervised from data that is unstructured or unlabeled. Also known as deep neural learning or deep neural network.

>Neural networks, also known as artificial neural networks (ANNs) or simulated neural networks (SNNs), are a subset of machine learning and are at the heart of deep learning algorithms. Their name and structure are inspired by the human brain, mimicking the way that biological neurons signal to one another.
 
>Artificial neural networks (ANNs) are comprised of a node layers, containing an input layer, one or more hidden layers, and an output layer. Each node, or artificial neuron, connects to another and has an associated weight and threshold. If the output of any individual node is above the specified threshold value, that node is activated, sending data to the next layer of the network. Otherwise, no data is passed along to the next layer of the network.

>Python :  In this model we have used the python programming language because it provides us with.    
●	Back- end and front- end development.
●	Cross platform language.    
●	Open source
●	Strong community base
●         Fewer and simple lines of code
                                                                                                                                                                                                                                                      
>Keras : Keras is a popular deep learning library that provides a user-friendly interface for building and training neural networks. It is built on top of other low-level libraries, such as TensorFlow or Theano, which handle the efficient computation of the underlying mathematical operations. Keras simplifies the process of designing, training, and deploying deep learning models, including those used for COVID-19 detection. 
The following are some of the advantages of using keras-
●	Keras helps in model construction using its intuitive syntax.
●	It provides a variety of pre- trained models some of which are mentioned in fig.
●	Data augmentation
●	Training and optimisation 
●	Integration with tensorflow
●	Deployment 
 
>Google Colab 
●Google Colab is a cloud-based platform provided by Google that allows users to write, run, and share Python code in a Jupyter Notebook environment. It is widely used in the deep learning community because it    provides free access to a GPU (Graphical Processing Unit) or TPU (Tensor Processing Unit), which can significantly accelerate                      
the training of deep learning models.
●Additionally, Google Colab integrates well with  popular deep learning frameworks like TensorFlow and PyTorch, making it easy to leverage their functionalities and pre-trained models in your projects.                 

#Approach towards the project:
>Uploading the dataset: First open a new notebook in Google colab  and then upload the dataset required for the model. We have downloaded a chest xray y-COVID 19 dataset from kaggle consisting of both positive and negative cases. The dataset includes train and test sets.The train set contains two folders -NORMAL and COVID with 300 images in each folder. The test set also contains two folders -NORMAL and COVID with 100 images in both folders. 

>Import the required libraries: At the beginning of your notebook, import the necessary libraries for data preprocessing. This typically includes libraries like TensorFlow, Keras, NumPy etc. 
NumPy:  is a Python library used for working with arrays.It stands for Numerical Python.
Pandas:  is an open-source library that allows you to perform data manipulation and analysis in Python.
Matplotlib : is a cross-platform, data visualization and graphical plotting library for Python and its numerical extension NumPy.

>Splitting the data:
In deep learning, the data is typically split into three main categories: training data, validation data, and test data. Each category serves a specific purpose in the development and evaluation of the model. Here's a breakdown of these categories:

  1. Training Data: The training data is the largest portion of the dataset and is used to train the deep learning model. It contains labeled examples (input data along with their corresponding target or output values) that the model uses to learn the patterns and relationships in the data. The model adjusts its parameters iteratively based on the training data to minimize the difference between its predicted outputs and the actual target outputs. The training data is crucial for the model to learn and improve its performance.

  2. Test Data: The test data is a completely independent dataset that is not used during the model training or hyperparameter tuning phases. It is used to evaluate the final performance of the trained model. The test data provides an unbiased assessment of how well the model generalizes to unseen examples. By using a separate test set, you can get a more accurate representation of the model's performance in real-world scenarios. The test data should be representative of the data the model will encounter in practice.



>Data Augmentation:
Data augmentation is a technique commonly used in deep learning to artificially increase the diversity of the training data by applying various transformations or perturbations to the existing data samples. It helps in improving the generalization and robustness of deep learning models. By creating new variations of the data, data augmentation can effectively increase the size of the training dataset and reduce overfitting It includes Image Transformations,Random Cropping and Padding,Zooming and Scaling,Rotation and Shearing,Noise Injection etc.

>Callbacks: A callback refers to a mechanism that allows you to monitor and control the training process of the model. Callbacks are commonly used in machine learning to perform certain actions at specific points during training, such as adjusting learning rates, saving model checkpoints, or early stopping.

>Sequential model:
The model is initialized as a sequential model, which means the layers are stacked on top of each other sequentially. The model consists of several layers. The first layer is a 2D convolutional layer (`Conv2D`) with 32 filters, a filter size of (5,5), ReLU activation function, and an input shape of (1000,1000,3), which indicates the size of the input image. After that, a max pooling layer (`MaxPooling2D`) with a pool size of (5,5) is added to downsample the features.This pattern is repeated with additional convolutional and max pooling layers, gradually increasing the number of filters (64 and 128) to capture more complex features.Next, a `Flatten` layer is added to flatten the multi-dimensional output into a vector. A fully connected layer (`Dense`) with 128 units and ReLU activation is added, followed by a `Dropout` layer with a dropout rate of 0.5. Dropout randomly removes a fraction of the connections between layers during training, which helps prevent overfitting. The final layer is another fully connected layer with 1 unit and sigmoid activation, which produces the output prediction. Another `Dropout` layer with a rate of 0.5 is added after this layer.

Then we use the fit method, which  is necessary to train a model, handle large datasets, specify the number of epochs, evaluate the model on validation data, and utilize callbacks for additional functionality during training.

>Output
-We created a dataframe called ' losses' to store and analyze loss values obtained during a machine learning task.Then we retrieved  the first few rows of the DataFrame losses, allowing us to quickly view the initial data entries related to the losses or errors obtained during a machine learning task.
Then we generate a line plot of the 'loss' and 'val_loss' columns from the losses DataFrame, allowing us to visualize and analyze the trend of the loss values during the training process.
Accuracy is a commonly used metric to evaluate the performance of a classification model. It measures the proportion of correctly predicted samples out of the total number of samples in a dataset
Then we used plot function to generate  a line plot of the. 'accuracy' and 'val_accuracy columns from the losses DataFrame, allowing us to visualize and analyze the trend of the accuracy values during the trainiing .
Similarly we generate a line plot of all the columns in the losses DataFrame, allowing us  to visualize and analyze the trends and variations of the values over time or across epochs.

-Predictions in a deep learning model refer to the model's output or estimated outcomes for a given input sample. They can represent class labels, probabilities, or continuous values depending on the type of problem the model is designed to solve. Predictions allow the model to make inferences and provide insights based on new, unseen data.
Then we create  a list called class_names with two elements representing the class names or labels for a binary classification problem related to COVID 19  detection. The list allows for easy access and manipulation of the class names in subsequent code or analysis.
Then we use the plot_prediction function , that generates a visualization of random predictions made by a model on a given test dataset. It plots the test images, customizes the title color based on the correctness of the predictions, and displays the plot using matplotlib.
 
-A histogram is a graphical representation of the distribution of a dataset. It provides a visual summary of the frequency or count of values within specific intervals, known as bins. The x-axis of the histogram represents the range of values in the dataset, while the y-axis represents the frequency or count of values falling within each bin.
-We utilize  matplotlib to plot a histogram . It sets the figure size, plots a bar chart, assigns axis labels and a plot title, saves the plot as an image, and displays the plot in the notebook.

-A confusion matrix is a table that is used to evaluate the performance of a classification model. It provides a summary of the predictions made by the model and compares them to the true labels of the data.It is typically represented as a square matrix, with the true labels as rows and the predicted labels as columns. The matrix provides a count of the number of samples that fall into each category (TP, TN, FP, FN). It allows for a detailed analysis of the model's performance, particularly in binary classification problems.
![image](https://github.com/Sinchana-SH/AI-model-to-predict-covid19/assets/116704673/35efb846-2611-4316-8866-d2634d2b81e4)

Using matplotlib we create a confusion matrix.It sets the figure size, font size, and color map, plots the confusion matrix, customizes the font size of the tick labels, sets the plot title, saves the plot as an image, and displays the plot in the notebook.

>Comparison between other Keras models
For the given dataset of COVID 19 chest rays, 4 models were trained, a sequential model, VGG19, VGG16 and  an Xception model.

➔The VGG16 model: which has 13 convolutional layers (convolutional blocks) followed by 3 fully connected layers.The number of filters and the size of the filters are gradually increased in each convolutional block. The spatial dimensions are reduced by max pooling. The model ends with a softmax activation in the output layer, which is suitable for multi-class classification tasks. The VGG16 model accuracy obtained is- 99.13%
 

➔The VGG19 model: It has a total of 16 convolutional layers (convolutional blocks) and 3 fully connected layers. The number of filters and the size of the filters increase in each convolutional block. The spatial dimensions are reduced by max pooling. The model ends with a softmax activation in the output layer, suitable for multi-class classification tasks. The VGG19 model accuracy obtained is- 89.99%
 

➔The Xception model:Consists of multiple separable convolutional blocks, which are made up of separable convolutions and pooling layers.
It gradually reduces the spatial dimensions while increasing the number of filters. The model ends with a global average pooling layer followed by fully connected layers. The output layer has a softmax activation function, suitable for multi-class classification tasks.The Xception model accuracy obtained is- 97.5%
 


In summary, the VGG16 and Xception models outperform the Sequential model, with VGG16 achieving the highest accuracy of 99.13%. VGG19 and Xception also show promising results, with accuracies of 89.99% and 97.5%, respectively. Therefore VGG16 or Xception provides better performance in image classification tasks.
> #Conclusion
At the end of this internship we were able to construct  4 deep learning models ,3 of which were Xception, VGG16 and VGG19 whose accuracy was 97.5% , 99.13% and   89.99% respectively. The last model was created by modifying an existing keras model. It had an accuracy of 89.13%.
The application of deep learning in COVID-19 radiologic image processing reduces false positives and negatives and offers a unique opportunity to provide fast, cheap, and safe diagnostic services to patients. The convolutional neural networks have demonstrated their potential and convinced themselves to be among the most prominent deep learning algorithms and powerful techniques in identifying anomalies, irregularities, and diagnostics in chest radiography. During a pandemic crisis, researchers focus on analyzing appropriate COVID-19 diagnoses by implementing CNN technology. The research revealed that using deep learning algorithms could enhance the detection features of X-Ray and CT scan images and the consciousness, accuracy, specificity, and efficiency of the diagnosis.
It had been a true learning experience as we were introduced to new tech stacks like google collab and kaggle. Coding and experimenting with different keras models made us understand on how different models have specific usage and accuracy. Overall it was a great start to exploration of AI-ML domain. 







NOTE-DATASETS:Test>>Normal+ Pneumonia
         Training set in 2 different folders-NORMAL+PNEUMONIA 
         
#RESULTS/PREDICTIONS:
>Sequential model: The green colour indicates the model prediction is correct,the red color indiactes model predicted it wrong.
![Screenshot (42)](https://user-images.githubusercontent.com/116704673/225034580-7ef2535b-c5cc-4818-9af4-a428b28a2100.png)

>PRETRAINED MODELS:Text with blue background indicates actual label and the text below the blue label is the predicted label,if has green text background it means its correctly predicted while red indicates wrong prediction.

>VGG16:

![image](https://github.com/Sinchana-SH/AI-model-to-predict-covid19/assets/116704673/bb3422c5-59ee-475e-8dd9-d322a22f5a96)

>VGG19

![image](https://github.com/Sinchana-SH/AI-model-to-predict-covid19/assets/116704673/e92975bc-541a-4f6d-a510-de3f5dab6c3d)

>XCEPTION

 ![image](https://github.com/Sinchana-SH/AI-model-to-predict-covid19/assets/116704673/041e2039-8a17-4405-950c-13ef2dad954f)

 

 
 Code Reference Credit:https://www.kaggle.com/code/hossamrizk/your-first-step-in-cnn (edited in some parts as per requirements)
 Dataset credit:https://www.kaggle.com/datasets/khoongweihao/covid19-xray-dataset-train-test-sets
