#encoding=utf8
'''
Detection with SSD
In this example, we will load a SSD model and use it to detect objects.
'''

import os
import sys
import argparse
import numpy as np
import cv2
import random

import caffe

from google.protobuf import text_format
from caffe.proto import caffe_pb2


detection_type = 'helmet'
# detection_type = 'head'
# detection_type = 'person'

def parse_args(detection_type):
    '''parse args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=1, help='gpu id')
    parser.add_argument('--data_shape', default=(540, 960), type=int)
    if detection_type == "person":
        parser.add_argument('--labelmap_file',
                            default=os.path.join(os.getcwd(), '../../', 'models/person_detection/person_labelmap_voc.prototxt'))
        parser.add_argument('--model_weights',
                            default=os.path.join(os.getcwd(), '../../', 'models/person_detection/pelee_ssd_person.caffemodel'))
        parser.add_argument('--model_def',
                           default=os.path.join(os.getcwd(), '../../', 'models/person_detection/pelee_ssd_person.prototxt'))
    elif detection_type == "helmet":
        parser.add_argument('--labelmap_file',
                            default=os.path.join(os.getcwd(), '../../', 'models/helmet/helmet_labelmap_voc.prototxt'))
        parser.add_argument('--model_weights',
                            default=os.path.join(os.getcwd(), '../../', 'models/helmet/pelee_helmet.caffemodel'))
        parser.add_argument('--model_def',
                           default=os.path.join(os.getcwd(), '../../', 'models/helmet/pelee_helmet.prototxt'))
    elif detection_type == "head":
        parser.add_argument('--labelmap_file',
                            default=os.path.join(os.getcwd(), '../../', 'models/head/head_labelmap_voc.prototxt'))
        parser.add_argument('--model_weights',
                            default=os.path.join(os.getcwd(), '../../', 'models/helmet/mobilev1-yolov3_head__iter_32000.caffemodel'))
        parser.add_argument('--model_def',
                           default=os.path.join(os.getcwd(), '../../', 'models/helmet/pelee_helmet.prototxt'))
    else:
        pass


    parser.add_argument('--image_dir', default=os.path.join(os.getcwd(), '../../', 'images/{}/'.format(detection_type)))
    parser.add_argument('--save_path', default=os.path.join(os.getcwd(), '../../', 'output/{}/'.format(detection_type)))
    return parser.parse_args()


def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in range(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

class CaffeDetection:
    def __init__(self, gpu_id, model_def, model_weights, labelmap_file, data_shape=None):
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()

        # Load the net in the test phase for inference, and configure input preprocessing.
        self.net = caffe.Net(model_def,      # defines the structure of the model
                             model_weights,  # contains the trained weights
                             caffe.TEST)     # use test mode (e.g., don't perform dropout)
         # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        if data_shape is None:
            data_shape = self.net.blobs['data'].data.shape
        else:
            if isinstance (data_shape, int):
                data_shape = (data_shape, data_shape)
        self.data_shape = data_shape    #(h, w)
        self.transformer = caffe.io.Transformer({'data': (1, 3, data_shape[0], data_shape[1])})
        # change into (C,H,W) instead of (H,W,C)
        self.transformer.set_transpose('data', (2, 0, 1))
        # self.transformer.set_mean('data', np.array([127.5, 127.5, 127.5])) # mean pixel
        self.transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
        # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_raw_scale('data', 255)
        #self.transformer.set_input_scale('data', 0.007843) ??????
        # the reference model has channels in BGR order instead of RGB
        self.transformer.set_channel_swap('data', (2, 1, 0))

        # load PASCAL VOC labels
        file = open(labelmap_file, 'r')
        self.labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(file.read()), self.labelmap)
        print('lablelmap:', self.labelmap)

    def detect(self, image_file, conf_thresh=0.5, topn=100):
        '''
        SSD detection
        '''
        # set net to batch size of 1
        # print(self.data_shape)
        # self.net.blobs['data'].reshape(1, 3, self.data_shape[0], self.data_shape[1])

        image = caffe.io.load_image(image_file)

        #Run the net and examine the top_k results
        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image

        # Forward pass.
        #detections = self.net.forward()['detection_out']

        self.net.forward()
        # detections = self.net.blobs['detection'].data[...]
        detections = self.net.blobs['detection_out'].data[...]
        print(detections.shape)
        # exit()
        # Parse the outputs.
        # print(detections[0,0,0,:])
        # print(detections[0,0,1,:])
        # exit()
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]

        # Get detections with confidence higher than conf_thresh.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]

        top_conf = det_conf[top_indices]
        print("top conf:", top_conf)
        top_label_indices = det_label[top_indices].tolist()
        # print('top_label_indices:', top_label_indices)
        top_labels = get_labelname(self.labelmap, top_label_indices)
        print(top_labels)
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        result = []
        for i in range(min(topn, top_conf.shape[0])):
            xmin = top_xmin[i]
            ymin = top_ymin[i]
            xmax = top_xmax[i]
            ymax = top_ymax[i]
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = top_labels[i]
            result.append([xmin, ymin, xmax, ymax, label, score, label_name])
        return result

def main(args):
    '''main '''
    detection = CaffeDetection(args.gpu_id,
                               args.model_def, args.model_weights,
                               args.labelmap_file, args.data_shape)
    #result = detection.detect(args.image_file)
    img_dir=args.image_dir
    save_path=args.save_path
    list_imgs=os.listdir(img_dir)
    for ind in list_imgs:
        imgPath=img_dir+ind
        if not os.path.isfile(imgPath):
            continue
        print(imgPath)
        result = detection.detect(imgPath) ##abspath
        img=cv2.imread(imgPath)	
        if img is None:
            continue
        imgshape=img.shape # h，w, c
        if imgshape[2]!=3:
            continue
        # cv2.imshow('fd',img)
        # cv2.waitKey(0)
        #print('shape')
        #print(imgshape)
        # colors = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        red = (0, 0, 255)
        blue = (255, 0, 0)
        for item in result:
            xmin = int(round(item[0] * imgshape[1]))
            ymin = int(round(item[1] * imgshape[0]))
            xmax = int(round(item[2] * imgshape[1]))
            ymax = int(round(item[3] * imgshape[0]))
            print('bbox[%f, %f, %f, %f]\n' % (xmin,ymin, xmax,ymax))
            if xmin<=0 or ymin<=0 or xmax<=0 or ymax<=0:
                print('Out of boundary.\n')
                continue
            if item[-1] == 'helmet':
                colors = blue
            elif item[-1] == 'face':
                colors = red
            cv2.rectangle(img, (round(xmin), round(ymin)),
                            (round(xmax), round(ymax)), colors, 2)
            # cv2.putText(img, '{:s} {:.3f}'.format(item[-1], item[-2]),
            #                    (round(xmin), round(ymin - 2)),
            #                    cv2.FONT_HERSHEY_SIMPLEX,
            #                    0.8,
            #                    colors,
            #                    2
            #                    )
            # crop_img = img[ymin:ymax,xmin:xmax]

        path_ = save_path + ind + '_detection.jpg'
        cv2.imwrite(path_, img)

if __name__ == '__main__':
    main(parse_args(detection_type))



