import face_recognition, cv2, json
import struct
from yolo_align import facealign
import numpy as np
def face_loc(model, known_face_encodings, known_face_names):
	#import libraries
	video_capture = cv2.VideoCapture('HM1.mp4')
	face_locations = []
	face_encodings = []
	face_names = []
	#To set the frame rate
	video_capture.set(3,510)
	video_capture.set(4,510)
	video_capture.set(5,45)
	process_this_frame = True
	#out = cv2.VideoWriter("output.avi", -1, 20, (640,480))

	fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
	out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (224,224))#(int(video_capture.get(3)),int(video_capture.get(4)))) 

	while True:
		#Start reading the frame
		ret, frame = video_capture.read()
		if ret==False:
			break
		frame = cv2.resize(frame, (224,224))
		# resize the frame by 0.25 to reduce calculataion
		#small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
		#rgb_small_frame = small_frame[:, :, ::-1]

		if process_this_frame:
			# gives the location of the face in terms of pixels
			face_locations, align_ims, img = facealign(frame, full=False)
			# gives the encodings of the frame image
			#print(align_ims)
			face_names = []
			#loop for saving mutiple face detection
			for ims in align_ims:
				print("Hi") 
				#ims = cv2.resize(ims,(224,224))
				print(ims.shape)
				face_encoding = img_to_encoding(ims, model)
				matches = compare_faces(known_face_encodings, face_encoding,tolerance = 0.2)
				# print((matches[0].shape))
				name = "Unknown"
				if True in matches:
					first_match_index = matches.index(True)
					name = known_face_names[first_match_index]

				face_names.append(name)
				print(face_locations)
				
		process_this_frame = not process_this_frame
		# open a text file to save the feature information

		for (left,top, right, bottom), name in zip(face_locations, face_names):
			#top *= 4
			#right *= 4
			#bottom *= 4
			#left *= 4

			# create a bounding rectangle to right name etc
			cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
			cv2.rectangle(frame, (left, bottom-35), (right, bottom), (255, 0, 0), cv2.FILLED)
			font = cv2.FONT_HERSHEY_DUPLEX
			cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
			
		i=0
		out.write(frame)
		#cv2.imshow('Video', frame)
		# if you press q then the loop breaks
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	video_capture.release()
	out.release()
	cv2.destroyAllWindows()

def img_to_encoding(image, model): 
    img = image[...,::-1]
    img = np.around(np.transpose(image, (1,0,2))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.test(x_train)
    return embedding

def compare_faces(known_en,face_en, tolerance=0.6):
	sum_sq = np.sum(np.linalg.norm(known_en - face_en, axis=1),axis=1)
	return list(sum_sq <= tolerance)
