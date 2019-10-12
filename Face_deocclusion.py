from __future__ import print_function
from collections import OrderedDict
import numpy as np
import cv2
from tqdm import tqdm
import os
import sys
import matplotlib.pyplot as plt
import pickle as pkl

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions

FACIAL_LANDMARKS_68_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("inner_mouth", (60, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])

FACIAL_LANDMARKS_5_IDXS = OrderedDict([
	("right_eye", (2, 3)),
	("left_eye", (0, 1)),
	("nose", (4))
])

# in order to support legacy code, we'll default the indexes to the
# 68-point model
FACIAL_LANDMARKS_IDXS = FACIAL_LANDMARKS_5_IDXS

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y

	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)

	# loop over all facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords

def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
	# create two copies of the input image -- one for the
	# overlay and one for the final output image
	overlay = image.copy()
	output = image.copy()

	# if the colors list is None, initialize it with a unique
	# color for each facial landmark region
	if colors is None:
		colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
			(168, 100, 168), (158, 163, 32),
			(163, 38, 32), (180, 42, 220)]

	# loop over the facial landmark regions individually
	for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
		# grab the (x, y)-coordinates associated with the
		# face landmark
		(j, k) = FACIAL_LANDMARKS_IDXS[name]
		pts = shape[j:k]

		# check if are supposed to draw the jawline
		if name == "jaw":
			# since the jawline is a non-enclosed facial region,
			# just draw lines between the (x, y)-coordinates
			for l in range(1, len(pts)):
				ptA = tuple(pts[l - 1])
				ptB = tuple(pts[l])
				cv2.line(overlay, ptA, ptB, colors[i], 2)

		# otherwise, compute the convex hull of the facial
		# landmark coordinates points and display it
		else:
			hull = cv2.convexHull(pts)
			cv2.drawContours(overlay, [hull], -1, colors[i], -1)

	# apply the transparent overlay
	cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

	# return the output image
	return output
 
class FaceAligner:
    def __init__(self, predictor, desiredLeftEye=(0.35, 0.35),
        desiredFaceWidth=256, desiredFaceHeight=None):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.predictor = predictor
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight
 
        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth
    def align(self, image, gray, rect):
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = self.predictor(gray, rect)
        shape = shape_to_np(shape)
 
        # extract the left and right eye (x, y)-coordinates
        (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]

        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
 
        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]
 
        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
            (leftEyeCenter[1] + rightEyeCenter[1]) // 2)
 
        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
 
        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h),
            flags=cv2.INTER_CUBIC)
 
        # return the aligned face
        return output

import imutils
import dlib
import cv2
import matplotlib.pyplot as plt

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=256)

def facealign(PATH):
	image = plt.imread(PATH)
	image = cv2.resize(image, (150,150))
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

	faceAligned = fa.align(image, gray, dlib.rectangle(0,0,150,150))
	#plt.imshow(faceAligned)
	#plt.show()
	#plt.pause(1)
	return faceAligned


def createDataMatrix(images):
	numImages = len(images)
	sz = images[0].shape
	data = np.zeros((numImages, sz[0] * sz[1] * sz[2]), dtype=np.float32)
	for i in range(0, numImages):
		image = images[i].flatten()
		data[i,:] = image
	return data

def readImages(path):
	images = []
	for filePath in tqdm(sorted(os.listdir(path))):
		fileExt = os.path.splitext(filePath)[1]
		if fileExt in [".jpg", ".jpeg"]:

			# Add to array of images
			imagePath = os.path.join(path, filePath)
			im = facealign(imagePath)

			if im is None :
				print("image:{} not read properly".format(imagePath))
			else :
				# Convert image to floating point
				im = np.float32(im)/255.0
				# Add image to list
				images.append(im)
				# Flip image 
				imFlip = cv2.flip(im, 1);
				# Append flipped image
				images.append(imFlip)
	numImages = int(len(images) / 2)
	# Exit if no image found
	if numImages == 0 :
		print("No images found")
		sys.exit(0)

	print(str(numImages) + " files read.")
	return images

# Add the weighted eigen faces to the mean face 
def createNewFace(*args):
	# Start with the mean image
	output = averageFace
	
	# Add the eigen faces with the weights
	for i in xrange(0,8):
		'''
		OpenCV does not allow slider values to be negative. 
		So we use weight = sliderValue - MAX_SLIDER_VALUE / 2
		''' 
		sliderValues[i] = cv2.getTrackbarPos("Weight" + str(i), "Trackbars")
		weight = sliderValues[i] - MAX_SLIDER_VALUE/2
		output = np.add(output, eigenFaces[i] * weight)

	# Display Result at 2x size
	output = cv2.resize(output, (0,0), fx=2, fy=2)
	cv2.imshow("Result", output)

def resetSliderValues(*args):
	for i in range(0, NUM_EIGEN_FACES):
		cv2.setTrackbarPos("Weight" + str(i), "Trackbars", int(MAX_SLIDER_VALUE/2));
	createNewFace()

NUM_EIGEN_FACES = 8
MAX_SLIDER_VALUE = 255
# dirName = "raw/HrithikRoshan"
# #dirName1 = "raw/MadhuriDixit"
# #dirName2 = "raw/Kajol"
# #dirName3 = "raw/Ali"
# images = readImages(dirName)
# # images = images+readImages(dirName1)
# # images = images+readImages(dirName2)
# # images = images+readImages(dirName3)
# sz = images[0].shape
# data = createDataMatrix(images)

# print("Calculating PCA ", end="...")
# mean, eigenVectors = cv2.PCACompute(data, mean=None, maxComponents=NUM_EIGEN_FACES)
# print(eigenVectors.shape)
# print ("DONE")

# averageFace = mean.reshape(sz)
with open("meanface.pkl","rb") as f:
	averageFace=pkl.load(f)
f.close()
meanface = averageFace

# rit = plt.imread("raw/HrithikRoshan/HirtikRoshan_198.jpg")
# plt.imshow()
# plt.show()

# eigenFaces = []
# for eigenVector in eigenVectors:
#   eigenFace = eigenVector.reshape(sz)
#   eigenFaces.append(eigenFace)
al = facealign("AMI.jpeg")

with open("eigenlist.pkl","rb") as f:
	eigenFaces=pkl.load(f)
f.close()
reconlist = []
w=0.45
v = [eigen.flatten() for eigen in eigenFaces]
m = averageFace.flatten()
x = al.flatten()
x12 = m#np.zeros(m.shape)
for vi in v:
  x12 = x12 + np.dot(vi,x-m)*vi
x11 = (w*x + (1-w)*x12)#.astype(int)
reconlist.append(x11.reshape(al.shape))
r = cv2.resize(reconlist[0], (224,224))
plt.imshow(r.astype(int))	
plt.pause(0)


# eigenFaces.pop(4)
# al = readImages("FACED")[0]
# reconlist = []

# x = al.flatten()
# x11 = averageFace.flatten()
# m = x11
# m = averageFace.flatten()
# x11 = x
# for vi in [eigen.flatten() for eigen in eigenFaces]:
#   x11 = x11 + 5*np.dot(vi,x-m)*vi
# reconlist.append(x11.reshape(al.shape))
# plt.imshow(averageFace)
# plt.show()
# plt.imshow(reconlist[0])
# plt.show()
# Create window for displaying Mean Face
# cv2.namedWindow("Result", cv2.WINDOW_AUTOSIZE)
     
# # Display result at 2x size
# output = cv2.resize(meanface, (0,0), fx=2, fy=2)
# cv2.imshow("Result", output)
 
# # Create Window for trackbars
# cv2.namedWindow("Trackbars", cv2.WINDOW_AUTOSIZE)
 
# sliderValues = []
     
# # Create Trackbars
# for i in xrange(0, NUM_EIGEN_FACES):
#   sliderValues.append(MAX_SLIDER_VALUE/2)
#   cv2.createTrackbar( "Weight" + str(i), "Trackbars", MAX_SLIDER_VALUE/2, MAX_SLIDER_VALUE, createNewFace)
     
#     # You can reset the sliders by clicking on the mean image.
# cv2.setMouseCallback("Result", resetSliderValues)
     
# print('''Usage:
# Change the weights using the sliders
# Click on the result window to reset sliders
# Hit ESC to terminate program.''')
 
# cv2.waitKey(0)
# cv2.destroyAllWindows()