#This code function returns face with landmarks list and landmarks being pointed out with circles
'''
def faceDetectedMat(imagePath,HAAR_DETECTOR_PATH, PREDICTOR_PATH):
    import numpy as np
    import cv2
    import dlib
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    face_cascade = cv2.CascadeClassifier(HAAR_DETECTOR_PATH)
    img = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
    if len(img.shape)==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(img, 1.3, 5)
    x,y,w,h = face[0]
    rect = dlib.rectangle(int(x),int(y),int(x+w),int(y+h))
    landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rect).parts()])
    for i in range(0,len(landmarks)):
            cv2.putText(img, str(i), (landmarks[i,0], landmarks[i,1]),fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,fontScale=0.4,color=(0, 0, 0))
            cv2.circle(img, (landmarks[i,0], landmarks[i,1]), 5, color=(255,255,255))
    return img,np.array(landmarks).tolist()
'''
def faceDetectedMat(imagePath,HAAR_DETECTOR_PATH, PREDICTOR_PATH):
    import numpy as np
    import cv2
    import dlib
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    face_cascade = cv2.CascadeClassifier(HAAR_DETECTOR_PATH)
    img = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
    if len(img.shape)==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(img, 1.3, 5)
    #print(face)
    if type(face)==np.ndarray:
        x = face[0][0]
        y = face[0][1]
        w = face[0][2]
        h = face[0][3]
        rect = img[y:y+h, x:x+w]
        rect = cv2.resize(rect, (154,154))
        #print(rect.shape)
        return rect
    else:
        return None
