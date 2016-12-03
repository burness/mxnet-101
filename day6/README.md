## pretrained model + finetuning
Usually, we have only small dataset, Training from Scratch seem to be poor performance.
Change the fc to be new class num layer:
As shown in [How to finetune mxnet model with a different net architecture? \#1313](https://github.com/dmlc/mxnet/issues/1313)
"If a layer has different dimensions, it should have a different name, then it will be initialized by default initializer 
instead of loaded."

Here, I have an example with wikiart []() using the pretrained inception-v3 model to new tasks:  [MXnet下Pretrained+Finetuning的正确姿势
](http://hacker.duanshishi.com/?p=1740)

## pretrained model + finetuning  and update the last layer's weights

