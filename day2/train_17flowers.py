import mxnet as mx
import argparse
import os, sys
import train_model
import importlib

def get_iterator(args, kv, data_shape=(3, 224, 224)):
    data_dir = args.data_dir
    train           = mx.io.ImageRecordIter(
        path_imgrec = os.path.join(args.data_dir, "train.rec"),
        mean_img = os.path.join(args.data_dir, "mean.bin"),
        data_shape = data_shape,
        batch_size = args.batch_size,
        rand_crop = True,
        rand_mirror = True,
        erbose = True,
        shuffle = True,
        num_parts = kv.num_workers,
        part_index = kv.rank
    )

    val = mx.io.ImageRecordIter(
        path_imgrec = os.path.join(args.data_dir, "test.rec"),
        mean_img    = os.path.join(args.data_dir, "mean.bin"),
        data_shape  = data_shape,
        batch_size  = args.batch_size,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)

    return (train, val)

def parse_args():
    parser = argparse.ArgumentParser(description='train an image classifer on mnist')
    parser.add_argument('--data-dir', type=str, default='17flowers/',
                        help='the input data directory')
    parser.add_argument('--network', type=str, default='alexnet',
                    help = 'the cnn to use')
    parser.add_argument('--gpus', type=str,
                        help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--num-examples', type=int, default=952,
                        help='the number of training examples')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='the batch size')
    parser.add_argument('--lr', type=float, default=.001,
                        help='the initial learning rate')
    parser.add_argument('--model-prefix', type=str,
                        help='the prefix of the model to load/save')
    parser.add_argument('--save-model-prefix', type=str,
                        help='the prefix of the model to save')
    parser.add_argument('--num-epochs', type=int, default=1000,
                        help='the number of training epochs')
    parser.add_argument('--load-epoch', type=int,
                        help="load the model on an epoch using the model-prefix")
    parser.add_argument('--lr-factor', type=float, default=1,
                        help='times the lr with a factor for every lr-factor-epoch epoch')
    parser.add_argument('--lr-factor-epoch', type=float, default=1,
                        help='the number of epoch to factor the lr, could be .5')
    parser.add_argument('--kv-store', type=str, default='local',
                        help='the kvstore type')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    net = importlib.import_module('symbol_' + args.network).get_symbol(17)
    kv = mx.kvstore.create(args.kv_store)
    # train
    train_model.fit(args, net, get_iterator)
