import os
import cv2
import mediapipe as mp
from preprocessing import mask, crop
import time
import numpy as np
import torch
import sys
from model import get_executor, transform

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# env variables
full_screen = False
WINDOW_NAME = 'Emotion Recognition Demo'
cam_height = 480 #720
cam_width = 640 #1280
cv2.resizeWindow(WINDOW_NAME, cam_width, cam_height)
cv2.moveWindow(WINDOW_NAME, 0, 0)
cv2.setWindowTitle(WINDOW_NAME, WINDOW_NAME)


# face/body
if len(sys.argv) < 2 or sys.argv[1] not in ['face', 'body']:
	print("First argument must be 'face' or 'body' as input modality!\nOptionally pass second argument as 'mask'.")
	quit()

modality = sys.argv[1]

# model
pretrained = "adjusted_mobilenet3_" + modality + ".pth"
if not os.path.isfile('pretrained/'+pretrained):
	if modality == 'face':
		os.system('gdown 1VeyoOrfcbxmAfJvJNMxSL967lja53_0X')
	else:
		os.system('gdown 14mtoYKoux-H_2S45MxHP5t3ZRoHYIj9E')
	os.system('mv '+pretrained+' pretrained/')

print('Using pretrained model: ' + pretrained)
model = get_executor('pretrained/'+pretrained)
model.eval()

AffectNet =["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Frustration", "Disgust", "Uncertainty"]

BoLD = ["Peace", "Affection", "Esteem", "Anticipation", "Engagement", "Confidence", "Happiness",
"Pleasure", "Excitement", "Surprise", "Sympathy", "Confusion", "Disconnect",
"Fatigue", "Embarrassment", "Yearning", "Disapproval", "Aversion", "Annoyance", "Anger",
"Sensitivity", "Sadness", "Disquietment", "Fear", "Pain", "Suffering"]

BoLD_dict = {}
positive = ["Happiness", "Affection", "Esteem", "Pleasure", "Excitement", "Sympathy", "Peace", "Engagement"]
neutral = ["Surprise", "Disconnect", "Yearning", "Sensitivity", "Confidence", "Confusion", "Anticipation"]
negative = ["Sadness", "Fatigue", "Pain", "Suffering", "Embarrassment", "Disquietment", "Disapproval", "Aversion", "Annoyance", "Anger", "Fear"]

for pos in positive:
	BoLD_dict[pos] = "Positive"
for neu in neutral:
	BoLD_dict[neu] = "Neutral"
for neg in negative:
	BoLD_dict[neg] = "Negative"

if modality == 'face':
	categories = AffectNet
elif modality == 'body':
	categories = BoLD


# demo
capture = cv2.VideoCapture(0)

if modality == 'face' or 'mask' in sys.argv:
	face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,
									  max_num_faces=1,
									  refine_landmarks=True,
									  min_detection_confidence=0.5)


# just initializing
prediction = 'Neutral'   
confidence = 0.5
preprocess_time = 0.0
inference = 0.01
i_frame = -1
while capture.isOpened():
	success, image = capture.read()
	if not success:
		break
	
	i_frame += 1
	
	with torch.no_grad():
		t1 = time.time()
		
		if modality == 'face' or 'mask' in sys.argv:
			results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
			if results.multi_face_landmarks:
				keypoints = results.multi_face_landmarks[0]
		
				if modality == 'face':
					(left, top, right, bottom) = crop(keypoints, image)
		
				if 'mask' in sys.argv:
					image = mask(keypoints, image)

				if modality == 'face':
					image = image[top:bottom, left:right]
			else:
				continue

		t2 = time.time()
		preprocess_time = t2-t1
	
		# prediction every 5 frames
		if i_frame % 5 == 0:	

			image = torch.from_numpy(transform(image)).float()
			out = model(image).numpy().reshape(-1)
			
			if modality == 'face':
				idx = np.argmax(out)
				confidence = out[idx]
				prediction = categories[idx]
				
			elif modality == 'body':

				positivity = sum([o for i, o in enumerate(out) if BoLD_dict[categories[i]] == 'Positive'])
				neutrality = sum([o for i, o in enumerate(out) if BoLD_dict[categories[i]] == 'Neutral'])
				negativity = sum([o for i, o in enumerate(out) if BoLD_dict[categories[i]] == 'Negative'])
				gen_cat = {"Positive": positivity, "Neutral": neutrality, "Negative": negativity}
				
				print(gen_cat)
				confidence = max(gen_cat.values())
				prediction = max(gen_cat, key=gen_cat.get)
			
			image = image.numpy()[0]
			image = np.transpose(image, axes=[1, 2, 0])
			t3 = time.time()
			inference = t3-t2
			
			inds = sorted(range(len(out)), key=lambda k: out[k])[::-1]
			print('########## Predictions ##########')
			for ind in inds:
				print(categories[ind], ':   \t', out[ind], sep='')
			print('\n')

		elif modality == 'body':
			image = transform(image).squeeze()
			image = np.transpose(image, axes=[1, 2, 0])

	image = cv2.resize(image, (cam_width, cam_height))
	image = image[:, ::-1]
	height, width, _ = image.shape
	
	label = np.zeros([height // 10, width, 3]).astype('uint8') + 255
	time_label = np.zeros([height // 10, width, 3]).astype('uint8') + 255
								
	cv2.putText(label, ' Prediction:  {}              Confidence: {:.2f}%'.format(prediction, confidence*100),
				(0, int(height / 16)),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.7, (0, 0, 0), 2)
					
	cv2.putText(time_label, ' Preprocessing Time: {:.2f} ms   Inference Time: {:.2f} ms   FPS: {:.2f}'.format(preprocess_time*1000, inference*1000, 1/(inference+preprocess_time)),
				(int(0), int(height / 16)),
				cv2.FONT_HERSHEY_SIMPLEX, 
				0.5, (0, 0, 255), 2)

	image = np.concatenate((time_label, image, label), axis=0)
	cv2.imshow(WINDOW_NAME, image)

	key = cv2.waitKey(1)
	if key & 0xFF == ord('q') or key == 27:  # exit
		break
		
		
capture.release()
cv2.destroyAllWindows()
