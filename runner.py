import cv2
import time
from multiprocessing import Pool, Process, Queue

from deepsort import demo_yolo3_deepsort
from VideoReader import launch_stream
from Window import manage_window
from config import VIDEO_PATH
from ProcessCreator import launch_process
from EventGraph import update_operations_value, update_operations_value_single, update_operations_value_traffic, make_TAG

import pickle as pk

stream_q = Queue()
batch_q = Queue()
results_q = Queue()
transfer_q = Queue()
video_q = Queue()
results = []

def print_results(results_q):
	count = 1
	while True:
		new = results_q.get()
		if new == "Done":
			print("All frames done.. exiting")
			break
		# new = [ process_single_frame(frame, ['left']) for frame in new ]
		count += 1
		# print("resu " , new[:2])
		results.append(new)


inter_q = Queue()

# def averger(inter_q):
# 	count = 1
# 	while True:
# 		new = inter_q.get()
# 		if new == "Done":
# 			print("All frames done.. exiting")
# 			break
# 		count += 1
# 		if count ==
# 		# print("resu " , new)
		# results.append(new)		


def create_video(video_q):
	a = dict()
	for _,j in results:
		a.update(j)
		
	c = {}
	for i in a:
		c[i] = set()
		c[i].add(a[i])

	for i in a:
		for j in range(5):
			if i+j in c:
				c[i+j].add(a[i])
			else:
				c[i+j] = set()
				c[i+j].add(a[i])
			if i-j in c:
				c[i-j].add(a[i])
			else:
				c[i-j] = set()
				c[i-j].add(a[i])
				
	fourcc =  cv2.VideoWriter_fourcc(*'MJPG')
	output = cv2.VideoWriter("demo.avi", fourcc, 20, (800, 600))
	count = 0
	while True:
		count += 1
		frame = video_q.get()
		print(frame)
		if frame == "Done":
			print("Video done...")
			break
		try:
			label = f'Swap{c[count]}'
			t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
			cv2.putText(frame, label ,(10, 10+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [0,0,0], 3)
		except KeyError:
			pass
		res = cv2.resize(frame, (800,600))
		self.output.write(res)


def Detector(stream_q, results_q):
	try:
		det = demo_yolo3_deepsort.Detector()
		det.area = 0, 0, 1920, 1080
		det.detect_from_stream(stream_q, results_q, video_q, create_video)
	except Exception as e:
		raise e from None
		print(e)

def batch_aggregator(batch_q, results_q):
	while True:
		new_batch = batch_q.get()
		print("---"*30)
		print("batch")
		print("---"*30)
		if new_batch == "Done":
			print("All done.. exiting")
			results_q.put("Done")
			break
		# processed_frames = [ process_single_frame(frame, ['left']) for frame in new_batch ]
		operators = ['front','front2','front3','front4','front5']
		operators.extend([f'front{i}' for i in range(6,26)])
		event_aggregated_graph = update_operations_value(new_batch, operators=operators)
		# make_TAG(new_batch, operators=operators )
		results_q.put(event_aggregated_graph)


launch_stream(VIDEO_PATH, stream_q)
launch_process(Detector, (stream_q, results_q, ), "Detector")
# launch_process(manage_window, (transfer_q, batch_q,), "Windower")
# launch_process(batch_aggregator, (batch_q, results_q,) , "Aggregator")
# launch_process(print_results, (batch_q,) , "Print results")
print_results(results_q)