import numpy as np
import cv2
from keras.models import load_model

model = load_model('model.h5')

# def get_image_size():
# 	img = cv2.imread('gestures/0/100.jpg', 0)
# 	return img.shape

image_x, image_y = 64,64
indices = [9,0,7,6,1,8,4,3,2,5]


def keras_process_image(img):
	img = cv2.resize(img, (image_x, image_y))
	img = np.array(img, dtype=np.float32)
	img = np.reshape(img, (1, image_x, image_y, 1))
	return img

def keras_predict(model, image):
	processed = keras_process_image(image)
	pred_probab = model.predict(processed)
	pred_class =indices[np.argmax(pred_probab)]
	print("predicted image is:",pred_class)
	return max(pred_probab), pred_class


def main():
	cap = cv2.VideoCapture(0)

	while(True):
    	# Capture frame-by-frame
    	ret, frame = cap.read()
   	
    	# Our operations on the frame come here
    	#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)COLOR_BGR2HSV
    	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    	#show_txt()
    	# Display the resulting frame
    	cv2.imshow('frame',gray)


    	keras_predict(model,frame)
    	
	
    	if cv2.waitKey(1) & 0xFF == ord('q'):
    	    break


def draw_circle(img,center,radius,color):
	cv.Circle(img, center, radius, color, thickness=1, lineType=8, shift=0) 

def put_splitted_text_in_blackboard(blackboard, splitted_text):
	y = 200
	for text in splitted_text:
		cv2.putText(blackboard, text, (4, y), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
		y += 50

def show_txt():
	blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
	splitted_text = split_sentence("hello world i am manish", 1)
	put_splitted_text_in_blackboard(blackboard, splitted_text)

def split_sentence(text, num_of_words):
	'''
	Splits a text into group of num_of_words
	'''
	list_words = text.split(" ")
	length = len(list_words)
	splitted_sentence = []
	b_index = 0
	e_index = num_of_words
	while length > 0:
		part = ""
		for word in list_words[b_index:e_index]:
			part = part + " " + word
		splitted_sentence.append(part)
		b_index += num_of_words
		e_index += num_of_words
		length -= num_of_words
	return splitted_sentence


main()