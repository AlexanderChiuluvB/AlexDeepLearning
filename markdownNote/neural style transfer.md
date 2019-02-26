neural style transfer

J(G) = aJ content (C, G) + βJ style (S, G)

* step1

  initialize the image G randomly

* step2

  use gradient descent to minimize J(G)

#### style cost function

***How correlated are the activations across different channels***

![1551019652211](/home/alex/.config/Typora/typora-user-images/1551019652211.png)

每个通道对应不同的神经元，用于识别图像不同的特性。如channel1对应识别图像是否有橙色的，而channel2对应识别图像是否有垂直的纹理，而不同通道之间的相关系数指的就是当图片某处出现这种垂直纹理，而该处又同时是橙色的可能性。



#### style matrix(Gram matrix)



![1551020196148](/home/alex/.config/Typora/typora-user-images/1551020196148.png)

style matrix



![1551020239189](/home/alex/.config/Typora/typora-user-images/1551020239189.png)

使用一个矩阵来表示不同filter之间的相关系数.i,j,k分别表示H,W,以及C通道数目。

![1551020295392](/home/alex/.config/Typora/typora-user-images/1551020295392.png)

生成图像和风格图像的风格矩阵都是这个形式，然后风格代价函数如下：

![1551020356122](/home/alex/.config/Typora/typora-user-images/1551020356122.png)



所以风格代价函数就是两个图像之间的范数再乘上一个归一化常熟。





