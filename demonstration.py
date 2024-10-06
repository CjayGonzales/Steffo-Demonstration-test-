import face_recognition
import cv2
import numpy as np

# Load the known images and get face encodings
known_image_1 = face_recognition.load_image_file("person1.jpg")
known_image_2 = face_recognition.load_image_file("person2.jpg")

# Get face encodings for known faces
known_encoding_1 = face_recognition.face_encodings(known_image_1)[0]
known_encoding_2 = face_recognition.face_encodings(known_image_2)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [known_encoding_1, known_encoding_2]
known_face_names = ["Person 1", "Person 2"]

# Load an unknown image
unknown_image = face_recognition.load_image_file("unknown.jpg")

# changes made


# more changes being made and commiting this comment to a new branch

# Find all the faces and face encodings in the unknown image
unknown_face_locations = face_recognition.face_locations(unknown_image)
unknown_face_encodings = face_recognition.face_encodings(unknown_image, unknown_face_locations)

# Convert the image to BGR for OpenCV to display it
image_to_display = cv2.cvtColor(unknown_image, cv2.COLOR_RGB2BGR)

# Loop through each face found in the unknown image
for (top, right, bottom, left), face_encoding in zip(unknown_face_locations, unknown_face_encodings):
    # See if the face is a match for any known face
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    # Use the known face with the smallest distance to the new face
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)

    if matches[best_match_index]:
        name = known_face_names[best_match_index]
    else:
        name = "Unknown"

    # Draw a box around the face
    cv2.rectangle(image_to_display, (left, top), (right, bottom), (0, 255, 0), 2)

    # Draw a label with the name below the face
    cv2.rectangle(image_to_display, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image_to_display, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

# Display the resulting image
cv2.imshow("Image", image_to_display)

# Wait for a key press and close the image window
cv2.waitKey(0)
cv2.destroyAllWindows()
