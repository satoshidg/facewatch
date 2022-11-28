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
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

            if face_results.detections:
                for detection in face_results.detections:
                    mp_drawing.draw_detection(image, detection)

            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    hands_detector.release()

def alertUser():
    play("sound.mp3", False)

if __name__ == '__main__':
    if sys.argv == "debug":
        DEBUG = True
    main()
