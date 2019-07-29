import cv2
import os
from multiprocessing import Pool, Process, Queue

def start_video_stream(path, put_queue):
	# vid = cv2.VideoCapture("deepsort/overtake.mp4")
	vid = cv2.VideoCapture()
	vid.open(path)
	try:
		# while vid.grab():
		# 	_, frame = vid.retrieve()
		while vid.isOpened():
			_, frame = vid.read()
			# print("pump ", frame.shape)
			gg=frame.shape
			# time.sleep(0.1)

			put_queue.put(frame[215:,463:])
	except AttributeError:
		put_queue.put("Done")
	except Exception as e:
		print(e, "in VideoStreamer")
		put_queue.put("Done")
	return


def frames_folder(path,put_queue):
	frames_name = os.listdir(path)
	frames_name.sort()
	for i in frames_name:
		frame = cv2.imread(path+i)[:,500:]

		put_queue.put(frame)
	put_queue.put("Done")


def launch_stream(path, put_queue):
	# streamer = Process(target=start_video_stream, args=(path, put_queue,)) # function for video stream
	streamer = Process(target=frames_folder, args=(path, put_queue,)) # function for photo stream
	streamer.name = "VideoStreamer"
	streamer.start()