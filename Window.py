from multiprocessing import Pool, Process, Queue
from config import BATCH_COUNT

## window manager that makes blocks of BATCH_COUNT and puts in batch_stream
def manage_window(input_stream, batch_stream):
	batch = []
	while True:
		new_frame = input_stream.get()
		# print("---"*30)
		# print("batch")
		# print("---"*30)
		if new_frame == "Done":
			# print("here da")
			if len(batch) != 0:
				print("Length of this batch - ", len(batch))
				batch_stream.put(batch)
			batch_stream.put("Done")
			break
		if len(batch) == BATCH_COUNT:
			print("Length of this batch - ", len(batch))
			batch_stream.put(batch[:])
			batch.clear()
		batch.append(new_frame)