import cv2
import mediapipe as mep
import time
import controller as cnt

time.sleep(1.0)
mep_hand = mep.solutions.hands
mep_drawing = mep.solutions.drawing_utils
ThumbCoordinates = (4,2)
FingerCoordinates = [(8,6),(16,14),(12,10),(20,18)]
video = cv2.VideoCapture(0)

with mep_hand.Hands(min_detection_confidence=0.8,min_tracking_confidence=0.8) as hands:
    while video.isOpened():
        ref, frame = video.read()
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = hands.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        print(results)
        if results.multi_hand_landmarks:
            handPoints = []
            for i, hand in enumerate(results.multi_hand_landmarks):
                mep_drawing.draw_landmarks(frame, hand, mep_hand.HAND_CONNECTIONS,mep_drawing.DrawingSpec(color=(23, 32, 42  ),thickness=3,circle_radius=5),
                                           mep_drawing.DrawingSpec(color=(255, 204, 0  ),thickness=2,circle_radius=4))
            for j, lm in enumerate(hand.landmark):
                #print(j,lm)
                height, width, color = frame.shape
                cx = int(lm.x* width)
                cy = int(lm.y*height)
                handPoints.append((cx,cy))
            count = 0
            for coordinate in FingerCoordinates:
                if handPoints[coordinate[0]][1]<handPoints[coordinate[1]][1]:
                    count +=1
            if handPoints[ThumbCoordinates[0]][0]<handPoints[ThumbCoordinates[1]][0]:
                count +=1

            cv2.putText(frame,str(count),(45,70),cv2.FONT_ITALIC,2,(23, 32, 42  ),3,cv2.LINE_AA)
            cnt.led(count)


        cv2.imshow('handtracking',frame)
        if cv2.waitKey(1)==ord('q'):
            break
video.release()
cv2.destroyAllWindows()



