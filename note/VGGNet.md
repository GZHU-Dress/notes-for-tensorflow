# VGGNet
## 神经结构
![](https://adeshpande3.github.io/assets/VGGNet.png)
C 相比 B 多了几个 1x1 的卷积层，其意义主要在于线性变换，而输入通道数和输出通道数不便，没有发生降维。

D, E 就是常说的 VGGNet-16 和 VGGNet-19。

VGGNet-16 主要分为六个不跟，前五个部分为卷积网络，最后一端是全连接网络。

![](http://www.pyimagesearch.com/wp-content/uploads/2017/03/imagenet_vgg16.png)
## 各级别网络参数量
|NetWork|A, A-LRN|B|C|D|E|
|:---|:---|:---|:---|:---|:---|
|Number of parameters|133|133|134|138|144|
*单位为百万*
