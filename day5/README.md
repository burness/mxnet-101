## Redefine eval metric
It's very easy to redefine the eval metric.

As shown in line \#86 in train_model.py.
Also you can define your own metric func. In [mxnet metric](https://github.com/dmlc/mxnet/blob/master/python/mxnet/metric.py), 
you can define the eval func