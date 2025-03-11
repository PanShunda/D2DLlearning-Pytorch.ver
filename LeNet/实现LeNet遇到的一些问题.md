## 实现LeNet遇到的一些问题

- **尝试瞎改变量名被制裁**

​	具体报错信息为：

```
    loss = loss(y_pred, y)
           ^^^^^^^^^^^^^^^
TypeError: 'Tensor' object is not callable
```

​	为什么呢？具体就是使用loss来接收loss()重复了，将loss改成其他的变量名即可。

- **使用了动画组件但是没有显示动画**

  这是**Matplotlib 后端配置问题**

  如果代码在 **非交互式环境**（如某些Jupyter Notebook配置或命令行脚本）中运行，Matplotlib可能默认使用不支持实时更新的后端（如 `Agg`），导致动画无法显示。

  解决方法：

  1.需导入matplotlib库

  ```
  import matplotlib
  matplotlib.use('TkAgg')  # 使用交互式后端
  import matplotlib.pyplot as plt
  ```

  2.使用animator时，需要手动刷新

  ```python
                  animator.add(epoch + (i + 1) / num_batches,
                               (train_l, train_acc, None))
                  plt.pause(0.001)  # 手动刷新
  ```

  3.在训练时，需要将窗口保持打开

  ```python
  train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
  plt.show()  # 保持窗口打开
  ```

  

## 实现LeNet遇到的一些*启示*

```python
net.eval()
```

的含义是切换到评估模式，这样可以使dropout等训练层失效，提升预测精度。

```python
def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使用GPU计算模型在数据集上的精度。"""
    # 1. 模型切换到评估模式（关闭dropout等训练专用层）
    if isinstance(net, torch.nn.Module):
        net.eval()  # 评估模式，关闭训练相关行为
        # 2. 自动获取模型所在设备（如果未指定device）
        if not device:
            device = next(iter(net.parameters())).device  # 从模型参数中获取设备

    # 3. 初始化累加器：[正确样本数, 总样本数]
    metric = d2l.Accumulator(2)

    # 4. 遍历数据集
    for X, y in data_iter:
        # 5. 将数据移动到模型所在的设备（GPU/CPU）
        if isinstance(X, list):
            X = [x.to(device) for x in X]  # 处理多输入模型（例如BERT）
        else:
            X = X.to(device)
        y = y.to(device)

        # 6. 计算当前批次的正确样本数，累加结果
        metric.add(d2l.accuracy(net(X), y), y.numel())  # y.numel()是批次大小

    # 7. 返回总体精度：正确样本数 / 总样本数
    return metric[0] / metric[1]
```

