import os
import cv2
import numpy as np

from deepsort.YOLO3.detector import YOLO3
from deepsort.deep_sort import DeepSort
from deepsort.util import COLORS_10, draw_bboxes
from ProcessCreator import launch_process
# from runner import create_video
import pickle as pk

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import time

# import queue

class Detector(object):
    def __init__(self):
        self.vdo = cv2.VideoCapture()
        self.yolo3 = YOLO3("deepsort/YOLO3/cfg/yolo_v3.cfg","deepsort/YOLO3/yolov3.weights","deepsort/YOLO3/cfg/coco.names", is_xywh=True)
        self.deepsort = DeepSort("deepsort/deep/checkpoint/ckpt.t7")
        self.class_names = self.yolo3.class_names
        self.write_video = True
        self.count = 0
        fourcc =  cv2.VideoWriter_fourcc(*'MJPG')
        self.output = cv2.VideoWriter("demo.avi", fourcc, 20, (600, 500))

    def open(self, video_path):
        assert os.path.isfile(video_path), "Error: path error"
        self.vdo.open(video_path)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.area = 0, 0, self.im_width, self.im_height
        if self.write_video:
            fourcc =  cv2.VideoWriter_fourcc(*'MJPG')
            self.output = cv2.VideoWriter("demo.avi", fourcc, 20, (self.im_width,self.im_height))
        return self.vdo.isOpened()
        
    def detect(self):
        xmin, ymin, xmax, ymax = self.area
        while self.vdo.grab(): 
            start = time.time()
            _, ori_im = self.vdo.retrieve()
            im = ori_im[ymin:ymax, xmin:xmax, (2,1,0)]
            bbox_xywh, cls_conf, cls_ids = self.yolo3(im)
            # print(cls_ids)
            print(self.count)
            self.count += 1
            if bbox_xywh is not None:
                # mask = cls_ids==2
                # bbox_xywh = bbox_xywh[mask]
                bbox_xywh[:,3] *= 1.2
                # cls_conf = cls_conf[mask]
                outputs = self.deepsort.update(bbox_xywh, cls_conf, im)
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:,:4]
                    identities = outputs[:,-1]
                    # print(identities)
                    ori_im = draw_bboxes(ori_im, bbox_xyxy, identities, offset=(xmin,ymin))

            end = time.time()
            # print("time: {}s, fps: {}".format(end-start, 1/(end-start)))

            cv2.imshow("test", ori_im)
            cv2.waitKey(1)

            if self.write_video:
                self.output.write(ori_im)

    def detect_from_stream(self, stream_queue, results_stream, video_q, create_video):
        xmin, ymin, xmax, ymax = self.area
        count = 0
        print("starting")
        to_draw = []
        while True: 
            start = time.time()
            try:
                ori_im = stream_queue.get()
                if ori_im == "Done":
                    results_stream.put("Done")
                    video_q.put("Done")
                    # launch_process(create_video, (video_q, ), "Video Creator")
                    break
                # print("here")
            except Exception as e:
                print(e, "in deep sort tracker")

            print(type(ori_im), " otype")
            print(ori_im.shape, " otype shape")

            im = ori_im[ymin:ymax, xmin:xmax, (2,1,0)]
            
            print(type(im), " itype")
            print(im.shape, " itype shape")

            bbox_xywh, cls_conf, cls_ids = self.yolo3(im)

            # print((bbox_xywh, cls_conf))

            # print(cls_ids)
            if bbox_xywh is not None:
                mask = cls_ids==2     ###  2 for car class, comment this to retain all classes
                bbox_xywh = bbox_xywh[mask]    ##this
                cls_conf = cls_conf[mask]  ## this
                bbox_xywh[:,3] *= 1.2 
                outputs = self.deepsort.update(bbox_xywh, cls_conf, im)
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:,:4]
                    identities = outputs[:,-1]
                    # print(identities)
                    results_stream.put({'identities': identities, 'bbox_xyxy': bbox_xyxy})
                    # print(identities)
                    ori_im = draw_bboxes(ori_im, bbox_xyxy, identities, offset=(xmin,ymin))
                    # if count > 1 and count < 120:
                    #     to_draw.append((bbox_xyxy, identities))
                    
            # end = time.time()
            # print("time: {}s, fps: {}".format(end-start, 1/(end-start)))
            # ori_im = cv2.resize(ori_im, (800,600))
            count += 1
            # if True:
            #     label = f'{count}'
            #     t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
            #     cv2.putText(ori_im, label ,(10, 10+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [0,0,0], 3)

            # cv2.imshow("test", ori_im)
            # cv2.waitKey(1)
            # video_q.put(ori_im)
            # if self.write_video:
            #     try:
           
            

                # #     except KeyError:
                # #         pass

                # cv2.imshow("test", ori_im)
                # cv2.waitKey(1)
            # res = cv2.resize(ori_im, (600,500))
            # self.output.write(res)
            

            # if count in a.keys():
            #     cv2.imwrite("thisframe_"+str(a[count])+".jpg", ori_im)
            # if count > 120:
            #     save_file('bboxfollowsby',to_draw)
            #     print("done")
            #     break
         


def save_file(filename, file):
    with open(filename, 'wb') as f:
        pk.dump(file,f)

if __name__=="__main__":
    import sys
    if len(sys.argv) == 1:
        print("Usage: python demo_yolo3_deepsort.py [YOUR_VIDEO_PATH]")
    else:
        cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("test", 800,600)
        det = Detector()
        det.open(sys.argv[1])
        det.detect()
