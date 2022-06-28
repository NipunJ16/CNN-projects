from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier(r'E:\Machine Learning(Study)\CNN\Emotion Detection CNN\FER-2013\haarcascade_frontalface_default.xml')
classifier = load_model(r'E:\Machine Learning(Study)\CNN\Emotion Detection CNN\FER-2013\model.h5')
gender_classifier = load_model(r'E:\Machine Learning(Study)\CNN\Age, Gender Detector\model.h5')

emotion = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
gender_labels = ['Male', 'Female']

video = cv2.VideoCapture(0)

while True:
    _, frame = video.read()
    labels = []
    g_labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            # print(type(prediction))
            prediction_gender = gender_classifier.predict(roi)[0]
            label = emotion[prediction.argmax()]
            gender_label = gender_labels[prediction_gender.argmax()]
            final_label = label + ", " + gender_label
            label_position = (x, y)
            cv2.putText(frame, final_label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Emotion and Gender Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()