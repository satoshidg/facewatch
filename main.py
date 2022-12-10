import sys
import cv2
import mediapipe as mp
from playsound import playsound as play

DEBUG = False

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection

def main():
    cap = cv2.VideoCapture(0)

    hands_detector = mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    face_detector = mp_face_detection.FaceDetection(
        model_selection=0, 
        min_detection_confidence=0.5
    )

    hands_logged = False
    face_logged = False

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hand_results = hands_detector.process(image)
        face_results = face_detector.process(image)

        if DEBUG:
            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if face_results.detections:
                if not face_logged:
                    print(face_results.detections)
                    face_logged = True
                for detection in face_results.detections:
                    mp_drawing.draw_detection(image, detection)

            if hand_results.multi_hand_landmarks:
                if not hands_logged:
                    print(hand_results.multi_hand_landmarks)
                    hands_logged = True

                for hand_landmarks in hand_results.multi_hand_landmarks:
                    temp = hand_landmarks.landmark[0]
                    # print(dir(temp))
                    # print(type(temp))
                    # print(temp.x)
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))

        if not face_results.detections or not hand_results.multi_hand_landmarks:
            continue
        
        # List of face boundaries each element should be (x_left, x_right, y_top, y_bottom)
        faces = list()
        for detection in face_results.detections:
            box = detection.location_data.relative_bounding_box
            faces.append((box.xmin, box.xmin + box.width, box.ymin, box.ymin + box.height))
        print("faces:", faces)



        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    hands_detector.release()

def debugPrint(message: str):
    if DEBUG:
        print(message)

def alertUser():
    play("sound.mp3", False)

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "debug":
        DEBUG = True
    main()
