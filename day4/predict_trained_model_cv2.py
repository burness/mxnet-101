import mxnet as mx
import logging
import numpy as np
import cv2
import scipy.io as sio

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
num_round = 260
prefix = "102flowers"
model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.cpu(), numpy_batch_size=1)
# synset = [l.strip() for l in open('Inception/synset.txt').readlines()]


def PreprocessImage(path, show_img=False):
    # load image
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean_img = mx.nd.load('mean.bin').values()[0].asnumpy()
    img = cv2.resize(img,(299,299))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img -mean_img
    img = img[np.newaxis, :]
    return img

right = 0
sum = 0
with open('test.lst', 'r') as fread:
    for line in fread.readlines():
        sum +=1
        batch  = '../day2/102flowers/' + line.split("\t")[2].strip("\n")
        batch = PreprocessImage(batch, False)
        prob = model.predict(batch)[0]
        # print prob 
        pred = np.argsort(prob)[::-1]
        # # Get top1 label
        # top1 = synset[pred[0]]
        top_1 = pred[0]
        if top_1 == int(line.split("\t")[1]):
            right_ratio = right/(1.0*(sum+0.000001))
            print right_ratio
            right += 1
print 'top 1 accuracy: %f '%(right/(1.0*sum))
