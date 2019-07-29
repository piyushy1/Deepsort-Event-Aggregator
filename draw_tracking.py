import cv2
from VideoReader import launch_stream
from multiprocessing import Queue
import numpy as np
import pickle as pk

with open("bboxfollowsby",'rb') as f:
	bbox = pk.load(f)

print(bbox)


frames_q = Queue()
PATH = '/home/piyush/tracking/Videos/follows_by/test/vid_9/9/'
launch_stream(PATH, frames_q)

frames = []

while True:
	new_frame = frames_q.get()
	if new_frame == "Done":
		print("Done")
		break
	frames.append(new_frame)

req_frames = frames[1:116]
new_frames = []


# colors = [(144,238,144),(178, 34, 34)]
colors =[(144,238,144),(178, 34, 34),(221,160,221),(  0,255,  0),(  0,128,  0),(210,105, 30),(220, 20, 60),
            (192,192,192),(255,228,196),( 50,205, 50),(139,  0,139),(100,149,237),(138, 43,226),(238,130,238),
            (255,  0,255),(  0,100,  0),(127,255,  0),(255,  0,255),(  0,  0,205),(255,140,  0),(255,239,213),
            (199, 21,133),(124,252,  0),(147,112,219),(106, 90,205),(176,196,222),( 65,105,225),(173,255, 47),
            (255, 20,147),(219,112,147),(186, 85,211),(199, 21,133),(148,  0,211),(255, 99, 71),(144,238,144),
            (255,255,  0),(230,230,250),(  0,  0,255),(128,128,  0),(189,183,107),(255,255,224),(128,128,128),
            (105,105,105),( 64,224,208),(205,133, 63),(  0,128,128),( 72,209,204),(139, 69, 19),(255,245,238),
            (250,240,230),(152,251,152),(  0,255,255),(135,206,235),(  0,191,255),(176,224,230),(  0,250,154),
            (245,255,250),(240,230,140),(245,222,179),(  0,139,139),(143,188,143),(255,  0,  0),(240,128,128),
            (102,205,170),( 60,179,113),( 46,139, 87),(165, 42, 42),(178, 34, 34),(175,238,238),(255,248,220),
            (218,165, 32),(255,250,240),(253,245,230),(244,164, 96),(210,105, 30)]


obs = [8,12,13]

for i in range(len(req_frames)):
	try:

		indices = { k:None for k in obs }
		for k in obs:
			indices[k] = np.where(bbox[i][1] == k)

		bboxes = { k:None for k in obs }
		for k in obs:
			bboxes[k] = bbox[i][0][indices[k]]

		pp = obs
		for l in range(len(obs)):
			x1,y1,x2,y2 = [int(i) for i in bboxes[pp[l]][0]]
			cv2.rectangle(req_frames[i], (x1, y1),(x2,y2), colors[l], 2)

		for j in range(i+1):
			indices = { k:None for k in obs }
			for k in obs:
				indices[k] = np.where(bbox[j][1] == k)
			# index_1 = np.where(bbox[j][1] == 1)
			# index_10 = np.where(bbox[j][1] == 10)
			bboxes = { k:None for k in obs }
			for k in obs:
				bboxes[k] = bbox[j][0][indices[k]]
			
			for l in range(len(obs)):
				x1,y1,x2,y2 = [int(i) for i in bboxes[ pp[l]][0] ]
				cv2.circle(req_frames[i], (  int((x1+x2)/2), int((y1+y2)/2) ), 2 ,colors[l], 2)
			# cv2.circle(req_frames[i], (  int((x12+x22)/2), int((y12+y22)/2) ), 2,colors[1], 2)

		new_frames.append(req_frames[i])
	except Exception as e:
		# raise e from None
		print(e)



	# cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 1, [255,255,255], 2)

shape = (440,480)
fourcc =  cv2.VideoWriter_fourcc(*'MJPG')
output = cv2.VideoWriter("bound.avi", fourcc, 20, shape)
for frame in new_frames:
	print(f'{frame.shape}')
	res = cv2.resize(frame, shape)
	output.write(res)


# def Detector(stream_q, results_q):
# 	try:
# 		det = demo_yolo3_deepsort.Detector()
# 		det.area = 0, 0, 1920, 1080
# 		det.detect_from_stream(stream_q, results_q)
# 	except Exception as e:
# 		raise e from None
# 		print(e)

# launch_process(Detector, (frames_q, transfer_q, ), "Detector")

