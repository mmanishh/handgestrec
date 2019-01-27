
# coding: utf-8

# In[19]:


import numpy as np
import cv2 as cv2
from keras.models import load_model
import pyttsx3


#for voice support
engine = pyttsx3.init()



# In[2]:


def say_sth(msg):
    engine.say(msg)


# In[3]:


face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

eye_cascade =cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')


# In[13]:

cap = None

model = load_model('model_rgb.h5')

image_x, image_y = 100,100

dict_labels = {
              0:'zero',
              1:'one',
              2:'two',
              3:'three',
              4:'four',
              5:'five',
              6:'six',
              7:'seven',
              8:'eight',
              9:'nine',
             10:'ten'
              }


# In[5]:


def keras_process_image(img):
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (1, image_x, image_y, 3))
    return img

def keras_predict(model, image):
    processed = keras_process_image(image)
    pred_probab = model.predict(processed)
    #print("pred_probab:",pred_probab,"\n")
    pred_class =np.argmax(pred_probab)
    return np.max(pred_probab), pred_class

def resize_save(image):
    resize = cv2.resize(image, (100, 100)) 
    cv2.imwrite("resized.jpg", resize)
    print("images saved")
    
def save_image(img,i):
    cv2.imwrite('images/zero/'+str(i)+'.jpg', img)
    print("images saved")


# In[6]:


def draw_circle(img,center,radius,color):
    cv.Circle(img, center, radius, color, thickness=1, lineType=8, shift=0) 

def put_splitted_text_in_blackboard(blackboard, splitted_text):
    y = 200
    for text in splitted_text:
        cv2.putText(blackboard, text, (4, y), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
        y += 50

def show_txt():
    blackboard = np.zeros((100, 400, 3), dtype=np.uint8)
    splitted_text = split_sentence("hello world i am manish", 1)
    put_splitted_text_in_blackboard(blackboard, splitted_text)
    cv2.imshow("Recognizing gesture", blackboard)

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

def destroy_window():
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


# In[16]:

def recognize():
    #defining size for rectangle
    x,y = 200,200
    w,h = 300,300
    threshold = 15
    
    pred_class = 10 #firstly intialized to be null
    prob =None  #variable declaration for probability
    global cap
    cap = cv2.VideoCapture(0)
    i = 0
    while(True):
        # Capture frame-by-frame
        ret, gray = cap.read()
        
        cropped_img = gray[x+threshold:x+w-threshold,y+threshold:y+h-threshold]
        # Our operations on the frame come here
        
        
        i+=1
        # Display the resulting frame
        cv2.rectangle(gray,(x,y),(x+w,y+h),(0,0,0),2)
        
        blackboard = np.zeros(gray.shape, dtype=np.uint8)
        
            
        if(i%10==0):
            global pred_class
            global prob
            prob, pred_class =keras_predict(model,cropped_img)
            # say_sth(dict_labels[pred_class])
            # engine.runAndWait()

        put_splitted_text_in_blackboard(blackboard,[str(dict_labels[pred_class]),'probability:',str(prob)])
        
        res = np.hstack((gray, blackboard))
        cv2.imshow('frame',res)
        
        keypress = cv2.waitKey(1)
        if keypress == ord('q'):
            destroy_window()
            break
        elif keypress == ord('c'):
            save_image(cropped_img,i) #if pressed c save the image 
    


# In[18]:


recognize()

# In[17]:


destroy_window()


# In[ ]:


#face detection code

#  faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#     for (x,y,w,h) in faces:
#         cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = gray[y:y+h, x:x+w]
#         print("====face detected========")
#         eyes = eye_cascade.detectMultiScale(roi_gray)
#         for (ex,ey,ew,eh) in eyes:
#             print("====eye detected========")
#             cv2.rectangle(roi_gray,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


# In[ ]:


# # Create a black image
# img = np.zeros((512,512,3), np.uint8)

# # Draw a diagonal blue line with thickness of 5 px
# cv2.line(img,(0,0),(511,511),(255,0,0),5)

# cv2.imshow("drawing shapes",img)

