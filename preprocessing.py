import numpy as np
import cv2

def crop(face_landmarks, image):
	left = int(min([face_landmarks.landmark[t].x for t in range(468)])*image.shape[1])
	top = int(min([face_landmarks.landmark[t].y for t in range(468)])*image.shape[0])
	right = int(max([face_landmarks.landmark[t].x for t in range(468)])*image.shape[1])
	bottom = int(max([face_landmarks.landmark[t].y for t in range(468)])*image.shape[0])
	expand_x = int(round(10/100 * (right-left)))
	expand_y = int(round(10/100 * (bottom-top)))
	left = max(0, left-expand_x)
	top = max(0, top-expand_y)
	right = min(right+expand_x, image.shape[1])
	bottom = min(bottom+expand_y, image.shape[0])
	return (left, top, right, bottom)


def mask(face_landmarks, image):

	# Jawline coordinates (more info in ./landmark_info/jawline_landmarks.txt)
	left_side = [234, 227, 116, 117, 118, 119, 47, 217]
	right_side = [454, 447, 345, 346, 347, 348, 277, 437]
	
	left_side_xs = [face_landmarks.landmark[t].x for t in left_side]
	right_side_xs = [face_landmarks.landmark[t].x for t in right_side]
	
	left_side_min = int(np.argmin(np.array(left_side_xs)))
	right_side_min = int(np.argmax(np.array(right_side_xs)))
					
	left_jawlines = [[234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152],
					[227, 137, 177, 215, 138, 135, 169, 170, 140, 171, 152],
					[116, 123, 147, 192, 210, 140, 171, 152],
					[117, 187, 214, 210, 211, 171, 152],
					[118, 50, 207, 212, 202, 32, 171, 152],
					[119, 101, 216, 57, 204, 32, 171, 152],
					[47, 142, 203, 92, 76, 106, 194, 208, 152],
					[217, 49, 165, 40, 91, 182, 208, 152]]
	
	right_jawlines = [[454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152],
					[447, 366, 401, 435, 367, 364, 394, 395, 369, 396, 152],
					[345, 352, 376, 416, 430, 369, 396, 152],
					[346, 411, 434, 430, 431, 396, 152],
					[347, 280, 427,	432, 422, 262, 396, 152],
					[348, 330, 436, 287, 424, 262, 396, 152],
					[277, 371, 423, 322, 306, 335, 418, 428, 152],
					[437, 279, 391, 270, 415, 406, 428, 152]]
	
	points = []

	left_jawline = left_jawlines[left_side_min]
	right_jawline = right_jawlines[right_side_min]
	
	left_jawline.extend(right_jawline[:-1][::-1])
	
	for j in left_jawline:
		x = face_landmarks.landmark[j].x
		y = face_landmarks.landmark[j].y
		y = int(y*image.shape[0])
		x = int(x*image.shape[1])
		point = (x, y)
		points.append(point)
		cv2.circle(image, (x, y), 2, (255, 255, 0), -1)

	# Nose coordinates
	y = int(face_landmarks.landmark[195].y*image.shape[0])
	x = int(face_landmarks.landmark[195].x*image.shape[1])
	mask_c = [((x), (y))]
	fmask_c = points + mask_c
	fmask_c = np.array(fmask_c, dtype=np.int32)
	cv2.polylines(image, [fmask_c], True, (255,255,255), thickness=4, lineType=cv2.LINE_8)
	
	# Fill mask						# cyan
	cv2.fillPoly(image, [fmask_c], (255,200,0), lineType=cv2.LINE_AA)
	
	return image
