![image-20260429203038944](C:\Users\10623\AppData\Roaming\Typora\typora-user-images\image-20260429203038944.png)

![image-20260429203051124](C:\Users\10623\AppData\Roaming\Typora\typora-user-images\image-20260429203051124.png)

![image-20260429203103366](C:\Users\10623\AppData\Roaming\Typora\typora-user-images\image-20260429203103366.png)

![image-20260429203111810](C:\Users\10623\AppData\Roaming\Typora\typora-user-images\image-20260429203111810.png)

![image-20260429203135372](C:\Users\10623\AppData\Roaming\Typora\typora-user-images\image-20260429203135372.png)

## 五组测试结果说明

- `test_scwssod_fusion_intersection.log`：先用 `fusion_intersection` 这套训练标签训练 `train.py`，再用 `test.py` 在固定 5 个公开测试集上评估得到。
- `test_scwssod_fusion_larger.log`：先用 `fusion_larger` 这套训练标签训练 `train.py`，再用 `test.py` 在固定 5 个公开测试集上评估得到。
- `test_scwssod_fusion_union.log`：先用 `fusion_union` 这套训练标签训练 `train.py`，再用 `test.py` 在固定 5 个公开测试集上评估得到。
- `test_scwssod_only_dot.log`：先用 `only_dot` 这套训练标签训练 `train.py`，再用 `test.py` 在固定 5 个公开测试集上评估得到。
- `test_scwssod_fusion_larger_weighted_e39.log`：这是在 `fusion_larger` 基线训练的基础上做的一个逐像素加权版本，标签语义仍然保持 `fusion_larger` 不变，即前景为 `1`、背景为 `2`、未知为 `0`；区别在于训练时额外读取了 `fusion_intersection`，把其中与 `fusion_larger` 前景重合的区域视为高置信前景，在 loss 里赋予更大的像素权重（默认 `2.0`），其余有效标注像素权重为 `1.0`、未知区域忽略不参与损失，因此它相当于只改了 `fusion_larger` 的监督强度分配，没有改原始伪标签本身，最后再加载可用的 `model-39.pt` 用 `test.py` 在固定 5 个公开测试集上评估得到。

 

没有交集0.95 其他1

随着交集扩散 与交集越近（越相似）的可信更高。 



纯净：全是交集  不纯净：不可靠，重新衡量 

权重函数：不纯净区域中属于目标像素与纯净像素的**中心**最近距离（更好的归一化方法？）、与纯净超像素相似颜色（RGB，CIElab衡量）中心区域均值相似度。 （需要知道谁主导？权重？）**[归一化]**

最终让那部分（除了纯净和未知和背景）的范围控制在0.9-1
