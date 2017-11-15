# -*- coding: utf-8 -*-

import time
import sys

class ProgressBar():
	def __init__(self, totalsum, info="", auto_display=True):
		self.totalsum = totalsum
		self.info = info
		self.finishsum = 0
		self.auto_display = auto_display
	def startjob(self):
		self.begin_time = time.time()
	def complete(self, num):
		self.gaptime = time.time() - self.begin_time
		self.finishsum += num
		if self.auto_display == True:
			self.display_progress_bar()
	def display_progress_bar(self):
		percent = float(self.finishsum) * 100 / self.totalsum
		eta_time = self.gaptime * 100 / percent - self.gaptime
		strprogress = "[" + "=" * int(percent / 2) + ">" + "-" * int(50 - percent / 2) + "]"
		str_log = ("%s %.2f %% %s %s/%s \t used:%d s eta:%d s" % (self.info, percent, strprogress, 
																 self.finishsum, self.totalsum, self.gaptime, eta_time))
		print '\r' + str_log

pb = ProgressBar(totalsum=100, info='progress')
pb.startjob()

for i in range(1000):
	time.sleep(0.1)
	pb.complete(1)
	pb.display_progress_bar()