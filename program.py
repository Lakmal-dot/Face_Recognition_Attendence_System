import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

# Define the directory where training photos are stored
training_dir = "Training_Photos"

# Initialize lists to store encodings and names
known_face_encoding = []
known_face_names = []

video_capture = cv2.VideoCapture(0)

# Iterate over each file in the training directory
for file_name in os.listdir(training_dir):
    # Check if the file is an image (you may want to add more checks here)
    if file_name.endswith((".jpg", ".jpeg", ".png",".jfif")):
        # Construct the full path to the image file
        image_path = os.path.join(training_dir, file_name)
        
        # Load the image using face_recognition library
        image = face_recognition.load_image_file(image_path)
        
        # Extract face encodings
        face_encodings = face_recognition.face_encodings(image)
        
        # Check if a face was found in the image
        if len(face_encodings) > 0:
            # Take the first face encoding (assuming only one face per image)
            face_encoding = face_encodings[0]
            
            # Extract the name from the file name (remove file extension)
            name = os.path.splitext(file_name)[0]
            
            # Append the face encoding and name to the respective lists
            known_face_encoding.append(face_encoding)
            known_face_names.append(name)

students = known_face_names.copy()

face_locations = []
face_encodings = []
face_names = []
s = True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date+'.csv','w+',newline='')
lnwriter = csv.writer(f)

while True:
    _,frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.face_distance(known_face_encoding,face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
            best_matches_index = np.argmin(face_distance)
            if matches[best_matches_index]:
                name = known_face_names[best_matches_index]

            face_names.append(name)
            if name in known_face_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name,current_time])

    cv2.imshow("Attendence system",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()



