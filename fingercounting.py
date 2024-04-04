import cv2
import time
import os
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)

wcam, hcam = 640, 480

cap.set(3, wcam)
cap.set(4, hcam)

folderpath = "Fingers"
mylist = os.listdir(folderpath)
fingerimages = []

for impath in mylist:
    image = cv2.imread(f'{folderpath}/{impath}')
    image = cv2.resize(image, (200, 200))
    fingerimages.append(image)

currentTime, previoustime = 0, 0

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
lmList = []

while True:

    ref, frame = cap.read()

    currentTime = time.time()
    fps = 1 / (currentTime - previoustime)
    previoustime = currentTime

    frame[0:200, 0:200] = np.ones((200,200,3),np.uint8)*255

    cv2.putText(frame, "FPS: " + str(int(fps)), (450, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    fingertips=[4,8,12,16,20]

    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

            fingers=[]


            if(lmList[fingertips[0]][1] > lmList[fingertips[0]-1][1]):
                
                fingers.append(1)

            else:
                fingers.append(0)

            for id in range(1,5):
                
                if(lmList[fingertips[id]][2] < lmList[fingertips[id]-2][2]):
                    fingers.append(1)

                else:
                    fingers.append(0)

            
            totalfingers=fingers.count(1)

            print(totalfingers)

            frame[0:200, 0:200]=fingerimages[totalfingers-1]

            cv2.rectangle(frame,(20,225),(170,425),(0,255,0),cv2.FILLED)

            cv2.putText(frame,str(totalfingers),(45,375),cv2.FONT_HERSHEY_PLAIN,10,(255,0,0),25)


    cv2.imshow("Finger Counting Project", frame)

    if cv2.waitKey(25) & 0xff == ord("q"):
        break
    
    lmList = []

cap.release()
cv2.destroyAllWindows()
