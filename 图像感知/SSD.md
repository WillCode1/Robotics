### YOLO

V1：https://www.jianshu.com/p/cad68ca85e27

网络结构很简单，最后一层使用线性激活函数。

输出是一个 7\*7\*30 的张量（tensor）

> 20个对象分类的概率
>
> 2个bounding box的位置（4x2）
>
> 2个bounding box的置信度

最后使用NMS遍历所有对象结果，输出预测的对象列表。

**问题**：如何标记bbox？

> 这里采用2个bounding box，有点不完全算监督算法，而是像**进化算法**。如果是监督算法，我们需要**事先**根据样本就能给出一个正确的bounding box作为回归的目标。



V2：https://www.jianshu.com/p/517a1b344a88

**改进**

> 全卷积网络
>
> 设置anchor box

输出是一个 13\*13\*125 的张量（tensor）

> 125=5*25
>
> 5个先验框
>
> 25，其中 T[:, :, :, :, 0:4] 为边界框的位置和大小(t_x, t_y, t_w, t_h)，T[:, :, :, :, 4] 为边界框的置信度，而 T[:, :, :, :, 5:] 为类别预测值



V3: https://zhuanlan.zhihu.com/p/36899263

均匀地密集采样、多尺度、锚点、FCN



### SSD

https://zhuanlan.zhihu.com/p/33544892

**设计理念**

**（1）采用多尺度特征图用于检测**

**（2）采用卷积进行检测**

与Yolo最后采用全连接层不同，SSD直接采用卷积对不同的特征图来进行提取检测结果。对于形状为 ![[公式]](https://www.zhihu.com/equation?tex=m%5Ctimes+n+%5Ctimes+p) 的特征图，只需要采用 ![[公式]](https://www.zhihu.com/equation?tex=3%5Ctimes+3+%5Ctimes+p) 这样比较小的卷积核得到检测值。

**（3）设置先验框**

**在Yolo中，每个单元预测多个边界框，但是其都是相对这个单元本身（正方块）**，但是真实目标的形状是多变的，Yolo需要在训练过程中自适应目标的形状。而SSD借鉴了Faster R-CNN中anchor的理念，每个单元设置尺度或者长宽比不同的先验框，预测的边界框（bounding boxes）是以这些先验框为基准的，**在一定程度上减少训练难度**。一般情况下，每个单元会设置多个先验框，其尺度和长宽比存在差异，如图5所示，可以看到每个单元使用了4个不同的先验框，图片中猫和狗分别采用最适合它们形状的先验框来进行训练，后面会详细讲解训练过程中的先验框匹配原则。

**网络结构**

![img](https://pic1.zhimg.com/80/v2-a43295a3e146008b2131b160eec09cd4_720w.jpg)



其中Priorbox是得到先验框，前面已经介绍了生成规则。检测值包含两个部分：类别**置信度和边界框**位置，各采用一次 ![[公式]](https://www.zhihu.com/equation?tex=3%5Ctimes3) 卷积来进行完成。令 ![[公式]](https://www.zhihu.com/equation?tex=n_k) 为该特征图所采用的先验框数目，那么类别置信度需要的卷积核数量为 ![[公式]](https://www.zhihu.com/equation?tex=n_k%5Ctimes+c) ，而边界框位置需要的卷积核数量为 ![[公式]](https://www.zhihu.com/equation?tex=n_k%5Ctimes+4) 。由于每个先验框都会预测一个边界框，所以SSD300一共可以预测
$$
38*38*4+19*19*6+10*10*6+5*5*6+3*3*4+1*1*4=8732
$$
个边界框（每个边界框c+4个值），这是一个相当庞大的数字，所以说SSD本质上是密集采样。