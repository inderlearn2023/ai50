
### What did you try?
I tried the following model parameter variations with 'relu', 'sigmoid' and 'tanh' for activation functions and 'softmax' as the output activation function.

![img_1.png](img_1.png)

### What worked well? 
Sigmoid activation function with 0.2 dropout had a better accuracy of 0.9808 with a loss of 0.0724
### What didn’t work well?
Activation function 'tanh' didn't seem to work for this data.
Also, dropout of 0.4 didn't work either.
### What did you notice?
For this dataset having multiple hidden layers didn't seem to improve the accuracy. Overall, sigmoid activation function worked well.
