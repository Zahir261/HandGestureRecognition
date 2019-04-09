# HandGestureRecognition

This project is based on Convolutional Neural network.

We selected 10 static gestures (Index, Peace, Three, Palm Opened, Palm Closed, OK, Thumbs, Fist, Swing, Smile) to recognize. Each class has 800 images for training purpose. The dataset is provided at https://drive.google.com/open?id=1uzbeBoJlVvFt9xCkxgKos6fD96pKcB26.

The CNN that we are going to use to recognize hand gesture is composed two convolution layer, two max pooling layer, two fully connected layer and output layer. There are three dropout performance in the network.

The first convolutional layer has 64 different layers with the kernel size 3x3. The activation function used in this layer is
Rectified Layer Unit (ReLU). As it is input layer, we have to specify the input size. The stride is set to default. The input shape is 50x50x1 which means that grayscale image of size 50x50 should be provided to this network. This layer produces the feature map and pass it to the next layer.

Then we have a max pooling layer with pool size 2x2 which takes the maximum value from a window of size 2x2. The spatial size of the representation is reduced progressively as the pooling layer takes only the maximum layer and discards the rest. This layer helps the network to understand the image better because it only selects more important features.

The next layer is another convolution layer and it has 64 different filters with the kernel size 3x3 and default stride. Again, ReLU was used as the activation function in this layer. This layer is followed by another max pooling layer which has a pooling size 2x2. In this layer we add our first dropout which randomly discards 25% of the total neurons to prevent the model from overfitting. Output from this layer is passed to the flatten layer.

Output from the previous layers are received by the flattening layer and they are flattened to a vector from two-dimensional matrix. This layer allows the fully connected layers to process the data we got till now.

The next layer is first fully connected layers which has 256 nodes and ReLU was used as the activation function. The layer is followed by dropout layer which excludes 25% of the neurons to prevent overfitting.

The second fully connected layer again has 256 nodes to receive the vector produced by first fully connected layer and uses ReLU as activation layer. The layer is followed by a dropout layer to exclude 25% of the neurons to prevent overfitting.

The output layer has 10 nodes corresponding to each classes of the hand gestures. This layer uses SoftMax function as activation layer which outputs a probabilistic value for each of the class.

The model is then compiled with Stochastic Gradient Descent (SGD) function with a learning rate 0.001. To evaluate loss, we used categorical cross entropy function since the model is compiled for more than two classes. Finally, the we specify the metrics of loss and accuracy to keep track on the evaluation process.
