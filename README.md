# 无约束优化算法——间接求解法（梯度法、牛顿法、共轭梯度法、阻尼牛顿法、变尺度法）

 “梯度法（Gradient_Descent）”

梯度法是求解无约束优化问题的解析法之一，又称为直接求解法。在理论上这个方法极为重要，因为它不仅提供了一个简单的、在一定场合令人满意的优化算法，而且许多更有效和实用的算法也常常是在这个基本算法基础上建立起来的，以时下火热的研究——神经网络技术为例，其中在神经网络参数更新的梯度反向传播过程中应用最广泛的算法之一就是随机梯度下降法（SGD），其在全局范围内的寻优能力强，且稳定性高的特点得到应用；基于梯度法衍生出的共轭梯度法等优化算法也有不同程度的改进和应用。
