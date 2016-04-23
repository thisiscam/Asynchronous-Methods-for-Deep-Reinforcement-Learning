from multiprocessing import Value, Lock, Semaphore

class SharedCounter(object):
    def __init__(self, initval=0):
        self.val = Value('i', initval)
        self.lock = Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    def value(self):
        with self.lock:
            return self.val.value

class Barrier:
    def __init__(self, n):
        self.n = n
        self.counter = SharedCounter(0)
        self.barrier = Semaphore(0)

    def wait(self):
        with self.counter.lock:
        	self.counter.val.value += 1
        	if self.counter.val.value == self.n: 
        		self.barrier.release()
        self.barrier.acquire()
        self.barrier.release()
