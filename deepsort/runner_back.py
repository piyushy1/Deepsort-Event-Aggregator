import demo_yolo3_deepsort
from multiprocessing import Pool, Process, Queue

import cv2

stream_q = Queue()
results_q = Queue()

results = []

def print_resu(results_q):
	while True:
		new = results_q.get()
		if new == "Done":
			print("50 frames done.. exiting")
			break
		print("resu " ,new)
		results.append(new)

# print("here")
import time
def pump(stream_q):
	vid = cv2.VideoCapture("overtake.mp4")
	count = 1
	try:
		while vid.isOpened():
			_, frame = vid.read()
			print("pump ", frame.shape)
			time.sleep(0.1)
			stream_q.put(frame)
			count += 1
	except Exception as e:
		stream_q.put("Done")
		print(e)

# def ano(stream_q, results_q):
# 	print("anot")
# 	while True:
# 		im = stream_q.get()
# 		results_q.put([im,(1,2,3)])

def ano2(stream_q, results_q):
	det = demo_yolo3_deepsort.Detector()
	det.area = 0, 0, 1920, 1080
	det.detect_from_stream(stream_q, results_q)

s = Process(target=pump, args=(stream_q, ))
s.name = "Pumper"
s.start()

h = Process(target= ano2, args=(stream_q, results_q, ))
h.name = "Another"
h.start()


p = Process(target= print_resu, args=(results_q,))
p.name = "This"
p.start()
# print_resu(results_q)