import face_recognition, cv2, glob, os
from yolo_align import facealign
import numpy as np
import tensorflow as tf

def face_encode(model):
	known_face_encodings = []
	known_face_names = []

	#run a loop on each photos kept in the images directory
	for name in glob.glob("images01/*"):
		for im_path in glob.glob(name+"/*"):
			print()
			a_image = cv2.imread(im_path)
			_, align_im, im = facealign(a_image, full=True)
			#align_im = np.expand_dims(align_im, 0)
			try:
				align_im = cv2.resize(align_im,(224,224))
			except:
				im= cv2.resize(im,(224,224))
				align_im = im
				#align_im=np.expand_dims(align_im,0)
			# create face encodings
			a_face_encoding=img_to_encoding(align_im, model)
			print(a_face_encoding.shape)
			# replace some words in name to write the image name
			name = name.replace('images/','')
			known_face_encodings.append(a_face_encoding)
			known_face_names.append(name)
	
	return known_face_encodings, known_face_names

def img_to_encoding(image, model): 
    img = image[...,::-1]
    img = np.around(np.transpose(img, (1,0,2))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.test(x_train)
    return embedding