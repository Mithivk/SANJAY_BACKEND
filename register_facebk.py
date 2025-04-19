import json
import face_recognition
import cv2
import os
import pickle

class FaceRecognition:
    def __init__(self):
        # Create a directory to store registered faces
        if not os.path.exists("registered_faces"):
            os.makedirs("registered_faces")
        # Load or initialize the known face encodings and names
        try:
            with open("registered_faces/face_encodings.pkl", "rb") as f:
                self.known_face_encodings, self.known_face_names = pickle.load(f)
        except FileNotFoundError:
            self.known_face_encodings = []
            self.known_face_names = []

        # Initialize the webcam


    def register_face(self, name, url):
        if url :
            self.video_capture = cv2.VideoCapture(url)
        else:
            return json.dumps({"message": "Please provide Video"})
        print(f"Registering face for {name}...")
        face_encodings = []
        total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame_num in range(total_frames):
            # Capture frame-by-frame
            ret, frame = self.video_capture.read()

            # Convert the image from BGR color (which OpenCV uses) to RGB color
            # rgb_frame = frame[:, :, ::-1]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to RGB
            # Find all the faces in the current frame
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            # Display the resulting image
            # cv2.imshow('Registering Face', frame)

            # If a face is found, break the loop
            if face_encodings:
                break

            # # Hit 'q' on the keyboard to quit!
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        # Save the face encoding and name
        self.known_face_encodings.append(face_encodings[0])
        self.known_face_names.append(name)

        # Save the updated face encodings to disk
        with open("registered_faces/face_encodings.pkl", "wb") as f:
            pickle.dump((self.known_face_encodings, self.known_face_names), f)

        print(f"Face registered for {name}!")

        # Release the webcam
        self.video_capture.release()
        cv2.destroyAllWindows()

    def recognize_face(self):
        # Load the known face encodings and names
        with open("registered_faces/face_encodings.pkl", "rb") as f:
            self.known_face_encodings, self.known_face_names = pickle.load(f)

        # Initialize the webcam
        video_capture = cv2.VideoCapture(0)
        while True:
            # Capture frame-by-frame
            ret, frame = video_capture.read()

            # Convert the image from BGR color (which OpenCV uses) to RGB color
            # rgb_frame = frame[:, :, ::-1]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to RGB
            # Find all the faces and face encodings in the current frame
            self.face_locations = face_recognition.face_locations(rgb_frame)
            self.face_encodings = face_recognition.face_encodings(rgb_frame, self.face_locations)

            # Loop through each face found in the frame
            for (top, right, bottom, left), self.face_encoding in zip(self.face_locations, self.face_encodings):
                # See if the face matches any known faces
                matches = face_recognition.compare_faces(self.known_face_encodings, self.face_encoding)
                name = "Unknown"

                # If a match was found in known_face_encodings, use the first one
                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]
                    return name

                # Draw a rectangle around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Display the resulting image
            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the webcam
        video_capture.release()
        cv2.destroyAllWindows()
# if __name__ == "__main__":
#     name = input("Enter the name of the person: ")
#     fr = Face_Recognition()
#     fr.register_face(name)