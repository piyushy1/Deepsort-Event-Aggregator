from multiprocessing import Process


### helper to launch process in the background

def cleaner(func, name):
	def new_func(*args):
		try:
			func(*args)
		except KeyboardInterrupt:
			print("Shutting down in ", name)
	return new_func

def launch_process(target, args, name=None):
	temp = Process(target=cleaner(target, name), args=args)
	temp.start()
