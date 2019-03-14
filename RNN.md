### RNN



![1552135054464](/home/alex/.config/Typora/typora-user-images/1552135054464.png)


$$
h_{t} = f_{W}(h_{t-1},x_t)
$$
$h_{t}​$ means states at time t,which means the current state is decided by both current input and previous state.

eg:
$$
h_{t} = tanh(W_{hh}h_{t-1}+W_{xh}x_{t})
\\
y_{t} = W_{hy}h_{t}
$$
at every time step we use the same weight matrix



![1552135523800](/home/alex/.config/Typora/typora-user-images/1552135523800.png)



h_i : hidden_state

x_i : input

y_i : output

L_i : losses L: final losses

f_w: weight ,for the whole process we use the same weight matrix



 #### Sequential Model



![1552135732203](/home/alex/.config/Typora/typora-user-images/1552135732203.png)



The encoder part can be seen as a many-to-one model,while the decoder part can be seen as a one-to many model



#### character-level language model

(supervised)



![1552135977457](/home/alex/.config/Typora/typora-user-images/1552135977457.png)



Each generated output character is decided by all the previous input characters and the current characters. Each time the output character is fed back in to the model.



#### Truncated backpropagation through time

So each forward propagation and backward propagation you need to go through the whole corpus seperately, which would be super slow and take ridiculously large memory.



![1552136660615](/home/alex/.config/Typora/typora-user-images/1552136660615.png)



somehow like mini-batch, each for/back propagation we will just focus on one batch of the corpus .

But we will using the same hidden states, like the first batch we have 6 hidden states, then we directly copy the 6 hidden states to the second batch,and then step forward and backward repeatedly.



#### Image Captioning



![1552138010628](/home/alex/.config/Typora/typora-user-images/1552138010628.png)



$Wih​$ is a 4096 dimension vector which describe the features of the image, then use the weight to feed the hidden layers.



![1552139098859](/home/alex/.config/Typora/typora-user-images/1552139098859.png)





Now CNN won't generate one vector to describe all the features at the end. Instead, it will

generate a grid vector, which only focus part of the image.And then it uses the partial weight and the input char to generate 2 outputs: distribution over the the vocabulary and the locations.



#### A problem



![1552141696889](/home/alex/.config/Typora/typora-user-images/1552141696889.png)



Each back pro the you have to multiply $W^{T}$， so if W>1 it will cause gradient exploding, and if W<1,it will cause gradient vanishing.



#### Solving Exploding gradients:

Gradient clipping if its norm is too big.

```
grad_norm = np.sum(grad*grad)
if grad_norm > threshold:
	grad *= (threshold/grad_norm)
```



#### Solving Vanishing gradients:



![1552142407542](/home/alex/.config/Typora/typora-user-images/1552142407542.png)



For the input,output and forget gate , we use sigmoid to make sure the output value falls between 0 and 1. 

so like if the if the value in forget gate is 0 ,then it will forget the corresponding value in previous cell.Otherwise it will still remember. SImilarly, the input gate use the similar mechanism to decide whether to write the g to cell.(The g is the ouput of the tanh funciton , which fall between -1 to 1)



![1552143300999](/home/alex/.config/Typora/typora-user-images/1552143300999.png)



#### Semantic Segmentation



![1552226403709](/home/alex/.config/Typora/typora-user-images/1552226403709.png)





![1552226603565](/home/alex/.config/Typora/typora-user-images/1552226603565.png)

So the basic idea is for each pixel we calculate a classification score, and the corresponding category that have the max score will be outputed.



![1552226527196](/home/alex/.config/Typora/typora-user-images/1552226527196.png)

* Downsampling: pooling and strided convolution 

* UpSampling: unpooling or strided transpose convolution



//看到Lecture 11 30:41







