# Student Performance Linear Regression Model

This is a linear regression model trained on student performance data to predict exam scores in math, reading, and writing based on several features. I go over the math and formulas in detail below. The dataset is from Kaggle, at https://www.kaggle.com/datasets/spscientist/students-performance-in-exams/data

This is a practice project I made to practice my machine learning skills. Below is my best effort to put what I've learned into words and describe the math behind the algorithm.

A lot of my learning thus far has come from The Hundred Page Machine Learning Book by Andriy Burkov and Stanfords CS229 Machine Learning course taught by Andrew Ng (available on YouTube).

## Linear Regression

Linear regression is a regression model. This means it predicts some real number between -Infinity and +Infinity. In this case, we train on several features that describe a student, and predict their exam scores in three subjects. The linear regression formula is:

```
y = wx + b
```

- y is the prediction, some real number between -Infinity and +Infinity.
- w is a vector of weights with the same dimension as x. Each weight cooresponds to the feature in the x vector that holds the same position.
- x is a vector of features. This dataset has 5 features: gender, race, parental education, lunch hour status, and whether or not test preperation was completed.
- b is the bias, a real number that represents the starting point of the line.

At its core, linear regression is just a linear line. `y = mx + b` is a linear line where m is the slope and b is the y-intercept. In this simple formula, each variable represents a real number. However, when linear regression is used, we often want to train on a dataset with many features, not just one. So, we modify this formula to use vectors, not just real numbers.

In our training dataset, we have 1,000 examples. Each example contains a feature vector (x) and a score (y). Linear regression's objective is to determine the best parameters (w, b) that create the best predictions. In other words, create a linear line that best fits the training data. To determine the accuracy of our predictions, we use a loss function.

## Loss Function

```
L = (1/N) * ∑(y-(wx + b))^2
```

This is a squared loss function. A loss function calculates how far off your predictions are from the real results in your training set. Here, we average the loss by summing over the loss of each example and dividing by N, the total number of examples. The goal of linear regression is to minimize loss. By minimizing loss, we ensure we are making the most accurate predictions. To minimize loss, we use an algorithm called gradient descent.

## Gradient Descent

Gradient descent is the algorithm used to minimize loss in linear regression. See the image below to visualize gradient descent.

<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/2c010918-b19b-4fdb-b2a8-61fcdbe88b1d" />

This graph represents the average loss of our training set, as calculated by the loss function in the prior step. With gradient descent, we start at the very top of the graph in the red section, and take steps towards the local minimum. With linear regression, there is only one minimum, meaning we are finding the global minimum. Once we reach the minimum, we know we have optimized our parameters to produce the lowest loss possible.

In order to take these steps towards the minimum, we need to take the partial derivates of the loss function with respect to w (the weights) and b (the bias). Each partial derivative is shown:

```
∂L/∂w = (1/N) * ∑-2x(y-(wx + b))

∂L/∂b = (1/N) * ∑-2(y-(wx + b))
```

To find these partial derivates, we use the chain rule. Now, with these partial derivates, we can begin the gradient descent algorithm. Gradient descent works by iteratively taking steps towards the minimum. We define the number of epochs, in this codebase is 15,000, which represents how many iterations we will perform of gradient descent.

In the first iteration, we initialize w to be the zero vector (<0, 0, 0, 0, 0>) and b to be 0. Then, in each iteration, we find the average of the partial derivates of loss with respect to w and b, and subtract them from the current w and b multiplied by alpha, our learning rate. Having a small learning rate, such as 0.001 in this codebase, ensures our algorithm takes small steps and does not overshoot the target. To find the partial derivate of the average of all training examples, we take the average of the sum of all partial derivates. You can see how this is all done in the `update_w_and_b` function in main.py.

The `train` function in main.py shows how we initialize w and b and iterate over each epoch. Once we have reached the number of epochs, we have our optimized weights and bias to plug into the linear regression formula. Now, when we want to predict some test score for a student with the feature vector <0, 2, 1, 5, 3>, we plug that vector in for x as well as the optimized w and b we found with gradient descent. That formula will produce y, the predicted test score for the student.

## Batch Gradient Descent vs Stochastic Gradient Descent

The form of gradient descent used here is referred to as batch gradient descent. This is because when finding the average partial derivative, we sum and evaluate the entire training set. While this training set is small with only 1,000 examples, performing this on a much larger dataset will be very slow. To solve this, stochastic gradient descent was invented to speed up the process. This form of gradient descent takes subsets of the training data, however, tends to follow a noisier path and is less straight than the batch version. This can be visualized below:

<img width="318" height="159" alt="image" src="https://github.com/user-attachments/assets/cb2b929c-c1d5-4196-8ee4-a5e430cf5062" />

## Outcome

Since this is not the largest dataset, the predictions made will not be the most accurate. However, the model does a decent job of making good predictions based on the data. In the image below, you can see the model training. As the epoch increases in each row, you can see the loss decreasing. This is what you want from a linear regression model. Since we are trying to optimize for the lowest loss possible, the loss should be decreasing with each iteration of gradient descent. Eventually, the parameters will be the most optimized they can be, and you'll see diminishing returns, as can be shown here.

<img width="1275" height="445" alt="image" src="https://github.com/user-attachments/assets/21184a3e-6921-4a84-b5aa-a4ec720d9b08" />

