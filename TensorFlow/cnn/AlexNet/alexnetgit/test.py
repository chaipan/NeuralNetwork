import TensorFlow.cnn.AlexNet.alexnetgit.caffe_classes as classes

import argparse
import sys
import urllib.request
import numpy as np
import os


def a():
	parser = argparse.ArgumentParser(description='Classify some images.')
	parser.add_argument('-m', '--mode', choices=['folder', 'url'], default='folder')
	parser.add_argument('-p', '--path', help='Specify a path [e.g. testModel]', default='testModel')
	args = parser.parse_args(sys.argv[1:])

	if args.mode == 'folder':
		# get testImage
		withPath = lambda f: '{}/{}'.format(args.path, f)
		# testImg = dict((f, cv2.imread(withPath(f))) for f in os.listdir(args.path) if os.path.isfile(withPath(f)))
	elif args.mode == 'url':
		def url2img(url):
			'''url to image'''
			resp = urllib.request.urlopen(url)
			image = np.asarray(bytearray(resp.read()), dtype="uint8")
			# image = cv2.imdecode(image, cv2.IMREAD_COLOR)
			return image

		# testImg = {args.path: url2img(args.path)}




if __name__ == "__main__":
	print("yes")
	# parser = argparse.ArgumentParser()
	# parser.parse_args()