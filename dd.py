import cv2
import os
import dlib
from keras.models import load_model
import numpy as np
from time import sleep
from pygame import mixer
import speech_recognition as sr
from playsound import playsound
from gtts import gTTS
from random import randint

mixer.init()
sound = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier('haar_cascade/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar_cascade/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar_cascade/haarcascade_righteye_2splits.xml')

PREDICTOR_PATH = "ad.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
#cascade_path='haarcascade_frontalface_default.xml'
#cascade = cv2.CascadeClassifier(cascade_path)
detector = dlib.get_frontal_face_detector()

def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def speak(text):
	tts=gTTS(text=text, lang="en")
	filename="voice.mp3"
	tts.save(filename)
	playsound(filename)

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def top_lip(landmarks):
    top_lip_pts = []
    for i in range(50,53):
        top_lip_pts.append(landmarks[i])
    for i in range(61,64):
        top_lip_pts.append(landmarks[i])
    top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    return int(top_lip_mean[:,1])

def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(65,68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56,59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
    return int(bottom_lip_mean[:,1])

def mouth_open(image):
    landmarks = get_landmarks(image)
    
    if landmarks == "error":
        return image, 0
    
    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return image_with_landmarks, lip_distance

yawns =0
yawn_status = False



lbl1='ho'
lbl2='hi'

model = load_model('models/cnnCat2.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
#count=0
score=0
thicc=2 
rpred=[99]
lpred=[99]

while(True):
    
    heartbeat = randint(50,75)
    
    ret, frame = cap.read()
    image_landmarks, lip_distance = mouth_open(frame)
    prev_yawn_status = yawn_status 

    if lip_distance > 25:
        yawn_status = True 
        
        cv2.putText(frame, "Subject is Yawning", (5000,450), 
                    cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)
        

        output_text = " Yawn Count: " + str(yawns + 1)

        cv2.putText(frame, output_text, (50,50),
                    cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
        
    else:
        yawn_status = False 
         
    if prev_yawn_status == True and yawn_status == False:
        yawns += 1

    cv2.imshow('Live Landmarks', image_landmarks )
    cv2.imshow('Yawn Detection', frame )
    #frame150 = rescale_frame(frame, percent=150)
   # cv2.imshow('frame150', frame150)
    
    height,width = frame.shape[:2] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    
    left_eye = leye.detectMultiScale(gray)
    
    right_eye =  reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0,height-50) , (800,height) , (0,0,0) , thickness=cv2.FILLED )

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
#        count=count+1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye= r_eye/255
        r_eye=  r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        rpred = model.predict_classes(r_eye)
        if(rpred[0]==1):
            lbl1='Open' 
        if(rpred[0]==0):
            lbl1='Closed'

    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
#        count=count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = model.predict_classes(l_eye)
        if(lpred[0]==1):
            lbl2='Open'   
        if(lpred[0]==0):
            lbl2='Closed'

    if(rpred[0]==0 and lpred[0]==0):
        score=score+2.5
        cv2.putText(frame,"both eyes Closed",(0,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
    elif(rpred[0]==1 and lpred[0]==0):
        score-=2.5
        cv2.putText(frame,"only right eye Closed",(0,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    elif(rpred[0]==0 and lpred[0]==1):
        score-=2.5
        cv2.putText(frame,"only left eye Closed",(0,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)

    else:
        score=score-2.5
        cv2.putText(frame,"Open",(0,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
        
    if(score<0):
        score=0   
    cv2.putText(frame,'Score:'+str(score),(400,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(frame,'Hb:'+str(heartbeat),(500,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
    if((score>20 and heartbeat<60) or yawns==2):
        yawns=0
        #person is feeling sleepy so we beep the alarm
        cnt=0
        while(cnt!=3):
            speak('hi its your assistant, tell me your name to make sure you are awake...answer it formally')
            
            r = sr.Recognizer()
            with sr.Microphone() as source:
                try:
                    audio = r.listen(source)
                except:
                    continue  
                said=''
                try:                
                    said=r.recognize_google(audio)
                except:
                    cnt=+1
                if(said==''):
                    cnt+=1
                print(said)
                
            if "name" in said:
                break
            else:
                cnt+=1
        
        sleep(2)
        while(cnt==3):
            sound.play()
            
            r = sr.Recognizer()
            with sr.Microphone() as source:
                try:
                    audio = r.listen(source)
                except:
                    continue    
                try:                
                    said=r.recognize_google(audio)
                except:
                    continue
                print(said)
            if(said):
                break;
            
        if(thicc<16):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        score=0
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
cap.release()
cv2.destroyAllWindows()


# sleep hb - 40-50
# normal hb: 72