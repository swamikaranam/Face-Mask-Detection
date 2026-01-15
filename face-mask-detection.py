import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("mask_detector_model.h5")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

print("[INFO] Starting real-time mask detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
   
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        
     
        face_resized = cv2.resize(face, (224, 224))
        face_array = np.array(face_resized, dtype="float32") / 255.0
        face_input = np.expand_dims(face_array, axis=0)

     
        preds = model.predict(face_input)[0]
        mask_prob = preds[0]
        without_mask_prob = preds[1]

       
        label = "Mask" if mask_prob > without_mask_prob else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)  # Green or Red

        
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

   
    cv2.imshow("Mask Detector", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
print("[INFO] Program ended.")