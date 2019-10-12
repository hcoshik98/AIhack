def face_encode(model):
	import face_recognition, cv2, glob, os
	from yolo_align import facealign
	from top10 import take10encodings
	10_face_encodings = []
	10_face_names = []
	#run a loop on each photos kept in the images directory
	for name in glob.glob("images/*"):
		known_face_encodings = []
		known_face_names = []
		for im_path in glob.glob(name+"/*"):
			a_image = cv2.imread(im_path)
			_, align_im, _ = facealign(a_image, full=True)
			# create face encodings
			a_face_encoding =model.get_layer("embeddings").output(align_im) 
			# replace some words in name to write the image name
			name = name.replace('images/','')
			known_face_encodings.append(a_face_encoding)
			known_face_names.append(name)

		10_face, 10_names = take10encodings(known_face_encodings, known_face_names)
		10_face_encodings.extend(10_face)
		10_face_names.extend(10_names)
		
	
	return 10_face_encodings, 10_face_names