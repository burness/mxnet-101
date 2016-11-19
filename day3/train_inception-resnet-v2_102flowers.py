import mxnet as mx
import argparse
import os, sys
import train_model

parser = argparse.ArgumentParser(description='train an image classifer on 102flowers')
parser.add_argument('--data-dir', type=str, default='./102flowers/',
                    help='the input data directory')
parser.add_argument('--gpus', type=str,
                    help='the gpus will be used, e.g "0,1,2,3"')
parser.add_argument('--num-examples', type=int, default=952,
                    help='the number of training examples')
parser.add_argument('--batch-size', type=int, default=16,
                    help='the batch size')
parser.add_argument('--data-shape', type=str, default='3,299,299',
                    help='the data shape, e.g "3,299,299"')
parser.add_argument('--lr', type=float, default=.001,
                    help='the initial learning rate')
parser.add_argument('--lr-factor', type=float, default=1,
                    help='times the lr with a factor for every lr-factor-epoch epoch')
parser.add_argument('--lr-factor-epoch', type=float, default=1,
                    help='the number of epoch to factor the lr, could be .5')
parser.add_argument('--model-prefix', type=str,
                    help='the prefix of the model to load')
parser.add_argument('--save-model-prefix', type=str,
                    help='the prefix of the model to save')
parser.add_argument('--num-epochs', type=int, default=700,
                    help='the number of training epochs')
parser.add_argument('--load-epoch', type=int,
                    help="load the model on an epoch using the model-prefix")
parser.add_argument('--kv-store', type=str, default='local',
                    help='the kvstore type')
args = parser.parse_args()

# download data if necessary
# def _download(data_dir):
#     if not os.path.isdir(data_dir):
#         os.system("mkdir " + data_dir)
#     os.chdir(data_dir)
#     if (not os.path.exists('train.rec')) or \
#        (not os.path.exists('test.rec')) :
#         import urllib, zipfile, glob
#         dirname = os.getcwd()
#         zippath = os.path.join(dirname, "17flowers.zip")
#         urllib.urlretrieve("http://data.dmlc.ml/mxnet/data/17flowers.zip", zippath)
#         zf = zipfile.ZipFile(zippath, "r")
#         zf.extractall()
#         zf.close()
#         os.remove(zippath)
#         for f in glob.glob(os.path.join(dirname, "102flowers", "*")):
#             name = f.split(os.path.sep)[-1]
#             os.rename(f, os.path.join(dirname, name))
#         os.rmdir(os.path.join(dirname, "17flowers"))
#     os.chdir("..")



# data
def get_iterator(args, kv):
    data_dir = args.data_dir
    # if Windows
    # if os.name == "nt":
    #     data_dir = data_dir[:-1] + "\\"
    # if '://' not in args.data_dir:
    #     _download(data_dir)
    data_shape = tuple([int(i) for i in args.data_shape.split(',')])
    train = mx.io.ImageRecordIter(
        path_imgrec = os.path.join(data_dir, "train.rec"),
        mean_img    = os.path.join(data_dir, "mean.bin"),
        data_shape  = data_shape,
        batch_size  = args.batch_size,
        rand_crop   = True,
        rand_mirror = True,
        shuffle = True,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)

    val = mx.io.ImageRecordIter(
        path_imgrec = os.path.join(data_dir, "test.rec"),
        mean_img    = os.path.join(data_dir, "mean.bin"),
        data_shape  = data_shape,
        batch_size  = args.batch_size,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)

    return (train, val)

if __name__ == '__main__':
    # train
    # network
    import importlib
    net = importlib.import_module('symbol_inception-resnet-v2').get_symbol(102)
    train_model.fit(args, net, get_iterator)
