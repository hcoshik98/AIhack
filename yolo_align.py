from __future__ import print_function
from collections import OrderedDict
import numpy as np
import cv2
from tqdm import tqdm
import os
import sys
import matplotlib.pyplot as plt
import pickle as pkl
import dlib

FACIAL_LANDMARKS_5_IDXS = OrderedDict([
	("right_eye", (2, 3)),
	("left_eye", (0, 1)),
	("nose", (4))
])
FACIAL_LANDMARKS_IDXS = FACIAL_LANDMARKS_5_IDXS

def shape_to_np(shape, dtype="int"):
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords

class FaceAligner:
	def __init__(self, predictor, desiredLeftEye=(0.35, 0.35),
		desiredFaceWidth=256, desiredFaceHeight=None):
		self.predictor = predictor
		self.desiredLeftEye = desiredLeftEye
		self.desiredFaceWidth = desiredFaceWidth
		self.desiredFaceHeight = desiredFaceHeight 
		if self.desiredFaceHeight is None:
			self.desiredFaceHeight = self.desiredFaceWidth
	def align(self, image, gray, rect):	
		shape = self.predictor(gray, rect)
		shape = shape_to_np(shape)
		(lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
		(rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
		leftEyePts = shape[lStart:lEnd]
		rightEyePts = shape[rStart:rEnd]
		leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
		rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
		dY = rightEyeCenter[1] - leftEyeCenter[1]
		dX = rightEyeCenter[0] - leftEyeCenter[0]
		angle = np.degrees(np.arctan2(dY, dX)) - 180
		desiredRightEyeX = 1.0 - self.desiredLeftEye[0]
		dist = np.sqrt((dX ** 2) + (dY ** 2))
		desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
		desiredDist *= self.desiredFaceWidth
		scale = desiredDist / dist
		eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
		    (leftEyeCenter[1] + rightEyeCenter[1]) // 2)
		M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
		tX = self.desiredFaceWidth * 0.5
		tY = self.desiredFaceHeight * self.desiredLeftEye[1]
		M[0, 2] += (tX - eyesCenter[0])
		M[1, 2] += (tY - eyesCenter[1])
		(w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
		output = cv2.warpAffine(image, M, (w, h),
		    flags=cv2.INTER_CUBIC)
		return output

def get_outputs_names(net):
	layers_names = net.getLayerNames()
	return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def refined_box(left, top, width, height):
	right = left + width
	bottom = top + height
	original_vert_height = bottom - top
	top = int(top + original_vert_height * 0.15)
	bottom = int(bottom - original_vert_height * 0.05)

	margin = ((bottom - top) - (right - left)) // 2
	left = left - margin if (bottom - top - right + left) % 2 == 0 else left - margin - 1

	right = right + margin

	return left, top, right, bottom

CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
IMG_WIDTH = 224
IMG_HEIGHT = 224
net = cv2.dnn.readNetFromDarknet('yolov3-face.cfg', 'yolov3-wider_16000.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def detect_face(frame):
	blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT), [0, 0, 0], 1, crop=False)
	net.setInput(blob)
	outs = net.forward(get_outputs_names(net))
	frame_height = frame.shape[0]
	frame_width = frame.shape[1]
	confidences = []
	boxes = []
	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			if confidence > CONF_THRESHOLD:
				center_x = int(detection[0] * frame_width)
				center_y = int(detection[1] * frame_height)
				width = int(detection[2] * frame_width)
				height = int(detection[3] * frame_height)
				left = int(center_x - width / 2)
				top = int(center_y - height / 2)
				confidences.append(float(confidence))
				boxes.append([left, top, width, height])
	indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
	refined_boxes = []
	for i in indices:
		i = i[0]
		box = boxes[i]
		left = box[0]
		top = box[1]
		width = box[2]
		height = box[3]
		left, top, right, bottom = refined_box(left, top, width, height)
		refined_boxes.append([left, top, right, bottom])
	return refined_boxes

predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=224)

def facealign(image, full=True):
	if full==True:
		try:
			image = cv2.resize(image, (224, 224))
		except:
			return [], image, image		
		bboxes = [[0,0,224,224]]
		gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		faceAligned = fa.align(image, gray, dlib.rectangle(0,0,224,224))
		return bboxes, faceAligned, image
	else:
		image = cv2.resize(image, (224, 224))
		bboxes = detect_face(image)
		gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		faceAligned = []
		for bb in bboxes:
			faceAligned.append(fa.align(image, gray, dlib.rectangle(bb[0],bb[1],bb[2],bb[3])))
		return bboxes, faceAligned, image

# def createDataMatrix(images):
# 	numImages = len(images)
# 	sz = images[0].shape
# 	data = np.zeros((numImages, sz[0] * sz[1] * sz[2]), dtype=np.float32)
# 	for i in range(0, numImages):
# 		image = images[i].flatten()
# 		data[i,:] = image
# 	return data

# def readImages(path):
# 	images = []
# 	for filePath in tqdm(sorted(os.listdir(path))):
# 		imagePath = os.path.join(path, filePath)
# 		im = facealign(imagePath)
# 		cv2.imwrite('./align_face/'+ filePath, im)
# 		im = np.float32(im)/255.0
# 		images.append(im)
# 		imFlip = cv2.flip(im, 1);
# 		images.append(imFlip)
# 	numImages = int(len(images) / 2)
# 	return images

# NUM_EIGEN_FACES = 15
# dirName = "./samples/"
# images = readImages(dirName)
# sz = images[0].shape
# data = createDataMatrix(images)

# print("Calculating PCA ", end="...")
# mean, eigenVectors = cv2.PCACompute(data, mean=None, maxComponents=NUM_EIGEN_FACES)
# print(eigenVectors.shape)
# print ("DONE")

# averageFace = mean.reshape(sz)
# with open("meanface.pkl","wb") as f:
# 	pkl.dump(averageFace, f)
# f.close()

# eigenFaces = []
# for eigenVector in eigenVectors:
# 	eigenFace = eigenVector.reshape(sz)
# 	eigenFaces.append(eigenFace)

# with open("eigenlist.pkl","wb") as f:
# 	pkl.dump(eigenFaces, f)
# f.close()
