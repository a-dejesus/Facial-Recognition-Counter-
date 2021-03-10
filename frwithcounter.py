import face_recognition
import cv2
from datetime import datetime, timedelta
import numpy as np
import platform
import pickle
import time
import serial
import os.path
from os import path 

# Our list of known face encodings and a matching list of metadata about each face.
known_face_encodings = []
known_face_metadata = []


serial_port = serial.Serial(  # open uart port to USB port
    port="/dev/ttyUSB0",
    baudrate=115200,
    bytesize=serial.EIGHTBITS, 
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,	#8N1 Protocol
    timeout=None
)

time.sleep(1)  # give it time to initialize


def UART_Receive():
    
    rval = serial_port.read()        # read from port
    val = int.from_bytes(rval,'big') # turn bytes into int(big endian) and store 
    return val                       # return received value
def UART_Transmit(counter):

    num = int(counter)               # convert into an integer
    sendval = bytes([num])           # convert into a byte
    serial_port.write(sendval)       # write a byte to pi

def activate():                      # class that responds to incoming flags from Raspberry pi
    while True:
        load_known_faces()           # loads known faces, has to execute because it's used in both paths
        print("Waiting for input from pi")
        val = UART_Receive()

        if val == 1:
            createprofilecounter()

        if val == 2:
            getprofilecounter()

def fcounter():                                                # fcounter creates and adds a counter to a text file
    if (path.exists('known_faces_counter.txt')==1):            # check if file exists
        with open("known_faces_counter.txt", "r") as fcount:   # open text file, with closes file once block executes
            nlist=[]                                           # create list for count

            for line in fcount:
                for char in line:
                    if char.isdigit():                         # parse through text file and check if value is a digit
                        nlist.append(char)                     # if it is, append to list

            count = "".join(nlist)                             # join list 
            count = int(count)                                 # change to int to modify
            count = count + 1                                  # add one
      
        with open("known_faces_counter.txt","w")as fcount:     # open file as write 
            fcount.write(str(count))                           # change count back to a string and write into txt file   
        with open("known_faces_counter.txt","r+")as fcount:
            reader2=fcount.read()                              # open file as read, read contents and return updated count
            print(reader2)
            return count


    if (path.exists('known_faces_counter.txt')==0):
        with open("known_faces_counter.txt", "w+") as fcount:  # this block makes the file and sets count to 0 if the file does not exist
            init=0                                               
            fcount.write(str(init))             
        with open("known_faces_counter.txt","r+")as fcount:
            reader2=fcount.read()
            print(reader2)
            return 0

def save_known_faces():
    with open("known_faces.dat", "wb") as face_data_file:
        face_data = [known_face_encodings, known_face_metadata]
        pickle.dump(face_data, face_data_file)
        print("Known faces backed up to disk.")
def load_known_faces():
    global known_face_encodings, known_face_metadata

    try:
        with open("known_faces.dat", "rb") as face_data_file:
            known_face_encodings, known_face_metadata = pickle.load(face_data_file)
            print("Known faces loaded from disk.")
    except FileNotFoundError as e:
        print("No previous face data found - starting with a blank known face list.")
        pass
def running_on_jetson_nano():
    # To make the same code work on a laptop or on a Jetson Nano, we'll detect when we are running on the Nano
    # so that we can access the camera correctly in that case.
    # On a normal Intel laptop, platform.machine() will be "x86_64" instead of "aarch64"
    return platform.machine() == "aarch64"
def get_jetson_gstreamer_source(capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=60, flip_method=0):
    
    #Return an OpenCV-compatible video source description that uses gstreamer to capture video from the camera on a Jetson Nano

    return (
            f"nvarguscamerasrc ! video/x-raw(memory:NVMM), " 
            f"width=(int){capture_width}, height=(int){capture_height}, " 
            f"format=(string)NV12, framerate=(fraction){framerate}/1 ! " 
            f"nvvidconv flip-method={flip_method} ! " 
            f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! " 
            "videoconvert ! video/x-raw, format=(string)BGR ! appsink"
            )
def register_new_face(face_encoding, face_image):
    
    #Add a new person to our list of known faces

    # Add the face encoding to the list of known faces
    known_face_encodings.append(face_encoding)
    # Add a matching dictionary entry to our metadata list.
    # We can use this to keep track of how many times a person has visited, when we last saw them, etc.
    known_face_metadata.append({

        "face_count": fcounter(),  # every time register new face is called, we update the counter and assign the new face an updated count
        "face_image": face_image,
    })
    
    
def lookup_known_face(face_encoding):
    
    #See if this is a face we already have in our face list
   
    metadata = None

    # If our known face list is empty, just return nothing since we can't possibly have seen this face.
    if len(known_face_encodings) == 0:
        return metadata

    # Calculate the face distance between the unknown face and every face on in our known face list
    # This will return a floating point number between 0.0 and 1.0 for each known face. The smaller the number,
    # the more similar that face was to the unknown face.
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

    # Get the known face that had the lowest distance (i.e. most similar) from the unknown face.
    best_match_index = np.argmin(face_distances)

    # If the face with the lowest distance had a distance under 0.6, we consider it a face match.
    # 0.6 comes from how the face recognition model was trained. It was trained to make sure pictures
    # of the same person always were less than 0.6 away from each other.
    # Here, we are loosening the threshold a little bit to 0.65 because it is unlikely that two very similar
    # people will come up to the door at the same time.
    if face_distances[best_match_index] < 0.65:
        # If we have a match, look up the metadata we've saved for it (like the first time we saw it, etc)
        metadata = known_face_metadata[best_match_index]  
  

    return metadata

 #function for brew

def getprofilecounter(): #retrieves counter matching profiles
    # Get access to the webcam. The method is different depending on if this is running on a laptop or a Jetson Nano.
   
    # Accessing the camera with OpenCV on a Jetson Nano requires gstreamer with a custom gstreamer source string
    video_capture = cv2.VideoCapture(get_jetson_gstreamer_source(), cv2.CAP_GSTREAMER)

    # Track how long since we last saved a copy of our known faces to disk as a backup.
    number_of_faces_since_save = 0

    while True:
        facedetect = 0
        # Grab a single frame of video
        ret, frame = video_capture.read()
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        # Find all the face locations and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        # Loop through each detected face and see if it is one we have seen before
        # If so, we'll give it a label that we'll draw on top of the video.
        face_labels = []
        for face_location, face_encoding in zip(face_locations, face_encodings):
            # See if this face is in our list of known faces.
            global metadata
            metadata = lookup_known_face(face_encoding)
            # If we found the face, label the face with some useful information.
            if metadata is not None:
                count = metadata["face_count"]                   # count for the face that is detected is called to transmit to pi
                UART_Transmit(count)
                print("profile number sent")
                facedetect = 1                                   # flag to break out of loop if we found a known face
                break
            break

        # Display the final frame of video 
        cv2.imshow('Video', frame)
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            save_known_faces()
            break

        # We need to save our known faces back to disk every so often in case something crashes.
        if len(face_locations) > 0 and number_of_faces_since_save > 100:
            save_known_faces()
            number_of_faces_since_save = 0
        else:
            number_of_faces_since_save += 1
        if facedetect ==1:
            break
    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

#function for create

def createprofilecounter(): #creates counter to match new profile
    # Accessing the camera with OpenCV on a Jetson Nano requires gstreamer with a custom gstreamer source string
    video_capture = cv2.VideoCapture(get_jetson_gstreamer_source(), cv2.CAP_GSTREAMER)
    # Track how long since we last saved a copy of our known faces to disk as a backup.
    number_of_faces_since_save = 0
    while True:
        # Grab a single frame of video
        facedetect = 0
        ret, frame = video_capture.read()
        try:
           # Resize frame of video to 1/4 size for faster face recognition processing
           small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        except:
           break
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all the face locations and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # Loop through each detected face and see if it is one we have seen before
        # If so, we'll give it a label that we'll draw on top of the video.
        face_labels = []
        for face_location, face_encoding in zip(face_locations, face_encodings):
            # See if this face is in our list of known faces.

            global metadata
            metadata = lookup_known_face(face_encoding)
            # If this is a brand new face, add it to our list of known faces
            if metadata is None:
                
                # Grab the image of the the face from the current frame of video
                top, right, bottom, left = face_location
                face_image = small_frame[top:bottom, left:right]
                face_image = cv2.resize(face_image, (150, 150))

                # Add the new face to our known face data
                register_new_face(face_encoding, face_image) # register new face and update counter 
                metadata = lookup_known_face(face_encoding)  # lookup known face since the new face is now in our known faces
                count = metadata["face_count"]               # retrieve the count value
                UART_Transmit(count)                         # transmit out counter over to our pi
                save_known_faces()                           # save known faces 

                facedetect = 1
                break #breaks us out of for loop
            break #breaks us out of for loop

        # Display the final frame of video 
        cv2.imshow('Video', frame)
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            save_known_faces()
            break

        # We need to save our known faces back to disk every so often in case something crashes.
        if len(face_locations) > 0 and number_of_faces_since_save > 100:
            save_known_faces()
            number_of_faces_since_save = 0
        else:
            number_of_faces_since_save += 1
        if facedetect == 1:
            break
    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

activate()   #start of program
