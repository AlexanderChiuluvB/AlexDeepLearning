# CS231N Course 1



## Data driven approach

### method to compare two images

You can compare two images pixel by pixel and add up all the differences

a reasonable choice for comparing may be **L1 Distance**


$$
d_1(l_1,l_2) = \sum_{p} \left |I^p_1-I^p_2 \right|
$$

```

### let say X is NXD where each row is an example
we wish to predict label for

distance = np.sum(np.abs(train-X[i,:]),axis=1)
```



other choices of distances like **Euclidean distance**


$$
d_2(I_1,I_2) = \sqrt{\sum_{p} \left(I^p_1-I^p_2\right)^2}
$$


or **Manhattan distance**


$$
d_2(I_1,I_2) = \sum_{p} |  I^P_1 - I^P_2 |
$$

### k- Nearest Neighbor Classifier

idea: instead of finding the single closest image in the training set,we will find the top k closest images, and have them vote on the label of the test image.

[![img](https://github.com/cs231n/cs231n.github.io/raw/master/assets/knn.jpeg)](https://github.com/cs231n/cs231n.github.io/blob/master/assets/knn.jpeg)



Notice that there are some green areas inside the blue area,which means inaccurate predictions.

While in 5-NN classifier, it will smooth over the irregularities,likely leading better generalization.



### Validation sets for **Hyperparameter tuning**

In task, try not to touch the test set, you can split the original training set into validation sets for hyper-parameter tuning

```

Evaluate on the test set only a single time,at the very end.

```



```
# assume we have Xtr_rows, Ytr, Xte_rows, Yte as before
# recall Xtr_rows is 50,000 x 3072 matrix
Xval_rows = Xtr_rows[:1000, :] # take first 1000 for validation
Yval = Ytr[:1000]
Xtr_rows = Xtr_rows[1000:, :] # keep last 49,000 for train
Ytr = Ytr[1000:]

# find hyperparameters that work best on the validation set
validation_accuracies = []
for k in [1, 3, 5, 10, 20, 50, 100]:
  
  # use a particular value of k and evaluation on validation data
  nn = NearestNeighbor()
  nn.train(Xtr_rows, Ytr)
  # here we assume a modified NearestNeighbor class that can take a k as input
  Yval_predict = nn.predict(Xval_rows, k = k)
  acc = np.mean(Yval_predict == Yval)
  print 'accuracy: %f' % (acc,)

  # keep track of what works on the validation set
  validation_accuracies.append((k, acc))
```



### Cross validation

[![img](https://github.com/cs231n/cs231n.github.io/raw/master/assets/crossval.jpeg)](https://github.com/cs231n/cs231n.github.io/blob/master/assets/crossval.jpeg)



Instead of choosing only one validation set, cross validation like 5 fold as above,will iterate over the choice of which fold is the validation fold,separately from 1-5.



### Pros and cons

* The classifier takes no time to train, since all that is required is to store and possibly index the training data. However, the test time is quite expensive.

* Pixel-based distances on high-dimensional data can be very unintuitive.

  [![img](https://github.com/cs231n/cs231n.github.io/raw/master/assets/samenorm.png)](https://github.com/cs231n/cs231n.github.io/blob/master/assets/samenorm.png)

​       as pics above L2 similarities are very different from perceptual similarities.

* very inefficient





### Parametric approach: Linear classifier



Linear classifier computes the score of a class as a weighted sum of all of its

pixel values across all 3 of its color channels. The function has the capacity to like

or dislike certain colors at certain positions in the image. 

> You might expect that the "ship" classifier would then have a lot of positive weights across its blue channel weights (presence of blue increases score of ship), and negative weights in the red/green channels (presence of red/green decreases the score of ship).

[![img](https://github.com/cs231n/cs231n.github.io/raw/master/assets/imagemap.jpg)](https://github.com/cs231n/cs231n.github.io/blob/master/assets/imagemap.jpg)



In the weight matrix, each row is a classifier for one of the classes,so you can  see the weight matrix as many classifier stacking together.

### Interpretation of linear classifiers as template matching

Each row of weight matrix corresponds to a template. The score of each class for an image is then obtained by comparing each template with the image using an *inner product*

Thinking in Nearest Neighbor way, now use the (negative) inner product as the distance instead of L1 or L2 distance.



### Bias Trick

[![img](https://github.com/cs231n/cs231n.github.io/raw/master/assets/wb.jpeg)](https://github.com/cs231n/cs231n.github.io/blob/master/assets/wb.jpeg)

### Image Data PreProcessing



* Normalization. It is important to center your data by subtracting the mean from every feature.



### Multiclass Support Vector Machine loss

（also called **hinge loss**）



$$
L_i = \sum_{j\neq y_i} \max(0,s_j-s_{y_i}+\Delta)
$$
```
def L_i_vectorized(x,y,W):
	scores = W.dot(x)
	margins = np.max(0,scores-scores[y]+1)
	margins [y] = 0
	return np.sum(margins)

```



> 
>
> why j!=y_i?  if j==yi,sj-s_yi = 0,then you will add up the loss 1

$$
L_i = \sum_{j\neq y_i} \max(0,s_j-s_{y_i}+\Delta)^2
$$

> what if we use square hinge loss? Sure,sometimes it can be seen and its performance is better than the hinge loss



example：

Lets unpack this with an example to see how it works. Suppose that we have three classes that receive the scores\( s = [13, -7, 11]\), and that the first class is the true class (i.e. \(y_i = 0\)). Also assume that \(\Delta\) (a hyperparameter we will go into more detail about soon) is 10. The expression above sums over all incorrect classes (\(j != y_i\)), so we get two terms:
$$
L_i = \max(0,-7-13+10)+\max(0,11-13+10)
$$


we are working with linear score functions:
$$
( ( f(x_i; W) = W x_i ) )
$$
so we can rewrite the loss function in this equivalent form:
$$
L_i = \sum_{j\neq y_i} \max(0, w_j^T x_i - w_{y_i}^T x_i + \Delta)
$$

```

```







### Weight Regularization

Suppose we have a dataset and a set of parameters W that correctly classify every example.

This set of parameters are not necessarily unique,so we need to find the most ideal parameters.

By extending the loss function with a regularization penalty that discourages large weights through an elementwise quadratic 
$$
R(W) = \sum_k\sum_l W_{k,l}^2
$$
#### motivation:

$$
x = [1,1,1,1]
\\
w_1 = [1,0,0,0]
\\
w_2 = [0.25,0.25,0.25,0.25]
$$

Which weight matrix is better?

sure it is w_2,since it takes each element in x into consideration.

while w_1 only focus on the first element.And L2 regularization aims to

make the weight more sparse,trying to consider most of the information in the x



In this expression above,we are summing up all the squared elements of W

hence the total loss is made up of data loss and regularization loss
$$
L =  \underbrace{ \frac{1}{N} \sum_i L_i }*\text{data loss} + \underbrace{ \lambda R(W) }*\text{regularization loss} \\
$$
or expand to:
$$
L = \frac{1}{N} \sum_i \sum_{j\neq y_i} \left[ \max(0, f(x_i; W)*j - f(x_i; W)*{y_i} + \Delta) \right] + \lambda \sum_k\sum_l W_{k,l}^2
$$
The most appealing property of L2 regularization is that it tends to improve generalization,

**because it means that no input dimension can have a very large influence on the scores all by itself**.



### Practical Consideration

* delta can always set to 1

* Relation to Binary support vector machine

  

* $$
  L_i = C \max(0, 1 - y_i w^Tx_i) + R(W)
  $$

here C is also a hyperparameter,it is related with lambda through reciprocal relation
$$
(C \propto \frac{1}{\lambda}).
$$



### Softmax Classifier

$$
L_i = -\log\left(\frac{e^{f_{y_i}}}{ \sum_j e^{f_j} }\right) \hspace{0.5in} \text{or equivalently} \hspace{0.5in} L_i = -f_{y_i} + \log\sum_j e^{f_j}
$$



softmax function:

用来代表第j类的的预测分数
$$
(f_j(z) = \frac{e^{z_j}}{\sum_k e^{z_k}} )
$$


Want to maximize the log likelihood, or (for loss function) to minimize the negative loss
$$
L_i = - log P(Y=y_i|X = x_i)
$$

* Q1 

maximum: infinite minimum:0(when it gives a score of 1 to one specific category)

* Q2

![1551156492352](C:\Users\AlexanderChiu\AppData\Roaming\Typora\typora-user-images\1551156492352.png)



probabilities would be 1/num_of_classes

so in coding you can check the loss after the first iteration and 1/num_of_classes ,in case making mistakes

**Information theory view**

The cross-entropy between a "true" distribution and an estimated distribution is defined as


$$
H(p,q) = - \sum_x p(x) \log q(x)
$$
The softmax classifier is hence minimizing the cross-entropy between the estimated class probabilities ans the true distribution.



**Probabilistic interpretation**
$$
P(y_i \mid x_i; W) = \frac{e^{f_{y_i}}}{\sum_j e^{f_j} }
$$
This can be interpreted as the normalized probability assigned to the correct label given the image and parameterized by W.

We are minimizing the negative log likelihood of the correct class,which can be interpreted as performing Maximum Likelihood Estimation.

### Numeric stability

the exponent terms can be very large! And dividing large number can be very unstable.


$$
\frac{e^{f_{y_i}}}{\sum_j e^{f_j}}
= \frac{Ce^{f_{y_i}}}{C\sum_j e^{f_j}}
= \frac{e^{f_{y_i} + \log C}}{\sum_j e^{f_j + \log C}}
$$

```

f = np.array([123, 456, 789]) # example with 3 classes and each having large scores
p = np.exp(f) / np.sum(np.exp(f)) # Bad: Numeric problem, potential blowup

# instead: first shift the values of f so that the highest number is 0:
f -= np.max(f) # f becomes [-666, -333, 0]
p = np.exp(f) / np.sum(np.exp(f)) # safe to do, gives the correct answer
```

A common choice of C would be 
$$
(\log C = -\max_j f_j )
$$





#### softmax 反向传播求导


$$
Loss = -\sum_{i}y_i * ln a_i
$$
suppose y_j = 1,then Loss becomes $Loss = -y_j*lna_j$ 


$$
\frac {\partial Loss}{\partial w_{ij}} = \frac{\partial Loss}{\partial a_i}*\frac{\partial a_i}{\partial z_i}*\frac{\partial z_i}{\partial w_{ij}}

= -\frac{1}{a_4}*\frac{\partial a_4}{\partial z_4}*1
$$
而当j==i的时候

![1551185104654](/home/alex/.config/Typora/typora-user-images/1551185104654.png)





当j!=i的时候



![1551185124438](/home/alex/.config/Typora/typora-user-images/1551185124438.png)



### some useful matrix differentiation rule

a is vector, x is vector,A is matrix
$$
\frac{d a^T x}{dx} = \frac{d x^T a}{dx} = a^T
$$

$$
\frac{dx^Tx}{dx} = 2x^T
$$

$$
\frac{d(x^Ta)^2}{dx} = 2x^Taa^T
$$

$$
\frac{dx^TA}{dx} = A^T
$$

$$
\frac {dx^TAx}{dx} = x^T(A+A^T)
$$

$$
\frac{\partial u^T}{\partial x} = (\frac{\partial u}{\partial x})^T
$$





example:


$$
E = \sum_{i=1}^{n} (d_{i}^Ta-v_i)^2 = || Da-v||^2
$$
prove:
$$
a = (D^TD)^-1D^Tv   
$$
minimizes error E



