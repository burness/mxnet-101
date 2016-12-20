import mxnet as mx
import logging
import numpy as np
import cv2
import scipy.io as sio
import time
# from tools.image_processing import resize, transform
import argparse
import os

CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Single-shot detection network video demo')
    parser.add_argument(
        '--video-file', dest='video_file', help='video file path', type=str)
    parser.add_argument(
        '--save-video',
        dest='save_vide',
        help='the path of the video to save',
        type=str)
    parser.add_argument(
        '--prefix',
        dest='prefix',
        help='trained model prefix',
        default=os.path.join(os.getcwd(), 'model', 'deploy_ssd_300'),
        type=str)
    parser.add_argument(
        '--epoch', help='epoch num of trained model', default=0, type=int)
    parser.add_argument(
        '--cpu',
        dest='cpu',
        help='(override GPU) use CPU to detect',
        action='store_true',
        default=False)
    parser.add_argument(
        '--gpu',
        dest='gpu_id',
        type=int,
        default=0,
        help='GPU device id to detect with')
    parser.add_argument(
        '--data-shape',
        dest='data_shape',
        type=int,
        default=300,
        help='set image shape')
    parser.add_argument(
        '--mean-r',
        dest='mean_r',
        type=float,
        default=123,
        help='red mean value')
    parser.add_argument(
        '--mean-g',
        dest='mean_g',
        type=float,
        default=117,
        help='green mean value')
    parser.add_argument(
        '--mean-b',
        dest='mean_b',
        type=float,
        default=104,
        help='blue mean value')
    parser.add_argument(
        '--thresh',
        dest='thresh',
        type=float,
        default=0.5,
        help='object visualize score threshold, default 0.6')
    parser.add_argument(
        '--nms',
        dest='nms_thresh',
        type=float,
        default=0.5,
        help='non-maximum suppression threshold, default 0.5')
    parser.add_argument(
        '--force',
        dest='force_nms',
        type=bool,
        default=True,
        help='force non-maximum suppression on different class')
    parser.add_argument(
        '--timer',
        dest='show_timer',
        type=bool,
        default=True,
        help='show detection time')
    args = parser.parse_args()
    return args


def visualize_detection_img(img, dets, spend_time, classes=[], thresh=0.6):
    import random
    height = img.shape[0]
    width = img.shape[1]
    colors = dict()
    # print dets.shape[0]
    cv2.putText(img, '{0} FPS'.format(1 / spend_time),
                (int(widht / 2 - 20), int(height / 2)), 1, 1,
                (random.randrange(0, 255), random.randrange(0, 255),
                 random.randrange(0, 255)), 1)
    for i in range(dets.shape[0]):
        cls_id = int(dets[i, 0])
        if cls_id >= 0:
            score = dets[i, 1]
            if score > thresh:
                if cls_id not in colors:
                    colors[cls_id] = (random.randrange(0, 255),
                                      random.randrange(0, 255),
                                      random.randrange(0, 255))
                xmin = int(dets[i, 2] * width)
                ymin = int(dets[i, 3] * height)
                xmax = int(dets[i, 4] * width)
                ymax = int(dets[i, 5] * height)
                cv2.rectangle(
                    img, (xmin, ymax), (xmax, ymin), color=colors[cls_id])
                class_name = str(cls_id)
                if classes and len(classes) > cls_id:
                    class_name = classes[cls_id]
                cv2.putText(img, '{:s} {:.3f}'.format(class_name, score),
                            (xmin, ymin - 2), 0, 0.4, colors[cls_id], 1)
    return img


if __name__ == '__main__':
    args = parse_args()
    if args.cpu:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args.gpu_id)
    predictor = mx.model.FeedForward.load(
        args.prefix, args.epoch, ctx=ctx, numpy_batch_size=1)

    cap = cv2.VideoCapture(args.video_file)
    ret, img = cap.read()
    img_shape = img.shape
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0,
                          (img_shape[1], img_shape[0]))

    while (cap.isOpened()):
        ret, img = cap.read()
        if ret:
            data_shape = (args.data_shape, args.data_shape)
            img2 = img.copy()
            img = cv2.resize(img, data_shape, interpolation=cv2.INTER_LINEAR)
            img_arr = np.asarray(img)
            img_arr = img_arr.copy()
            img_arr[:, :, (0, 1, 2)] = img_arr[:, :, (2, 1, 0)]
            img_arr = img_arr.astype(float)
            pixel_means = [args.mean_r, args.mean_g, args.mean_b]
            img_arr -= pixel_means
            channel_swap = (2, 0, 1)
            im_tensor = img_arr.transpose(channel_swap)
            im_tensor = im_tensor[np.newaxis, :]
            start = time.time()
            detections = predictor.predict(im_tensor)
            spend_time = time.time() - start
            det = detections[0, :, :]

            res = det[np.where(det[:, 0] >= 0)[0]]
            result = visualize_detection_img(
                img2, res, spend_time, CLASSES, thresh=0.5)
            out.write(result)
        else:
            break
    cv2.waitKey(0)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
