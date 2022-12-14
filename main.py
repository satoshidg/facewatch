import sys
import cv2
import mediapipe as mp
import time
import winsound
import platform


DEBUG = False
SYSTEM = platform.system()
THRESHOLD = 5

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection

def main():
    cap = cv2.VideoCapture(0)
    count = 0

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
        image.flags.writeable = DEBUG
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hand_results = hands_detector.process(image)
        face_results = face_detector.process(image)

        if DEBUG:
            displayFrame(image, face_results, hand_results)

        is_touching = determineTouching(face_results, hand_results)

        if is_touching:
            count += 1
            count = min(count, THRESHOLD)
            if count == THRESHOLD:
                alertUser()
        else:
            count -= 1
            count = max(0, count)
        
        time.sleep(1)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    hands_detector.release()


def alertUser():
    if SYSTEM == 'Windows':
        winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
    elif SYSTEM == 'Darwin':
        # TODO: implement whne Mac
        pass


def determineTouching(face_results, hand_results):
    if not(face_results.detections and hand_results.multi_hand_landmarks):
        return False
    
    # List of face boundaries each element should be (x_left, x_right, y_top, y_bottom)
    faces = list()
    for detection in face_results.detections:
        box = detection.location_data.relative_bounding_box
        faces.append((box.xmin, box.xmin + box.width, box.ymin, box.ymin + box.height))

    for hand in hand_results.multi_hand_landmarks:
        for landmark in hand.landmark:
            x = landmark.x
            y = landmark.y

            for face in faces:
                if face[0] <= x <= face[1] and face[2] <= y <= face[3]:
                    return True
    
    return False


def displayFrame(image, face_results, hand_results):
    # Draw the hand annotations on the image.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if face_results.detections:
        for detection in face_results.detections:
            mp_drawing.draw_detection(image, detection)

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "debug":
        DEBUG = True
    main()
