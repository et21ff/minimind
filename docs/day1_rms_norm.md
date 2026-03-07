# RMS_NORM

为什么我们需要RMS_NORM?

![20260307153200.png](https://raw.githubusercontent.com/et21ff/picbed/main/img/20260307153200.png)

梯度（一阶导）计算很大程度上由x决定，如果x过大梯度会爆炸 需要NORM

比起layernorm 不需要减去x拔 只需要方差标准化到1 即可
