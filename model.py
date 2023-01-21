import torch
import numpy as np
import cv2

def transform(frame: np.ndarray):
	# 480, 640, 3, 0 ~ 255
	frame = cv2.resize(frame, (224, 224))  # (224, 224, 3) 0 ~ 255
	frame = frame / 255.0  # (224, 224, 3) 0 ~ 1.0
	frame = np.transpose(frame, axes=[2, 0, 1])  # (3, 224, 224) 0 ~ 1.0
	frame = np.expand_dims(frame, axis=0)  # (1, 3, 224, 224) 0 ~ 1.0
	return frame

def get_executor(pretrained):
	
	model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
	if 'body' in pretrained:
		classes = 26
	else:
		classes = 8
	feature_dim = 1280
	
	modules = list(model.children())[:-1]  # delete the last fc layer.
	avg = torch.nn.AdaptiveAvgPool2d(output_size=(1,1))
	flat = torch.nn.Flatten()
	new_fc = torch.nn.Linear(feature_dim,classes)
	soft = torch.nn.Softmax(dim=1)
	modules.append(avg)
	modules.append(flat)
	modules.append(new_fc)
	modules.append(soft)
	model = torch.nn.Sequential(*modules)

	cp = torch.load(pretrained, map_location=torch.device('cpu'))['state_dict']			
	model.load_state_dict(cp,strict=True)
	return model