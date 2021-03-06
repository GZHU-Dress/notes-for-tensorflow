# AlexNet

--------------------------------------------------------------------------------

## AlexNet 主要使用到的新技术点

- 成功使用 ReLU 作为 CNN 的激活函数，并验证其效果在较深的网络抄过了 Sigmoid，成功解决了其在网络较深时的梯度弥散问题。
- 训练时使用 Dropout 随机忽略一部分神经元，以避免模型过拟合。
- 在 CNN 中使用重叠的最大池化。AlexNet 全部使用最大池化，避免平均池化的模糊化效果。并且提出让步长比池化核的尺寸小，这样池化层的输出之间会有重叠和覆盖，提升了特征的丰富性。
- 提出了LRN 层，对局部神经元的活动创建竞争机制，使得其中响应比较大的值变得相对更大，并抑制其他反馈较小的神经元，提升了特征的丰富性.(神经元极化)
- GPU 运算
- 数据增强，随机截取

  ## AlexNet 网络结构

  ![](https://adeshpande3.github.io/assets/AlexNet.png)
