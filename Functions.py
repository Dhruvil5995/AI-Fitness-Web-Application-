import cv2
import pickle
import numpy as np
import pandas as pd
import mediapipe as mp
from flask import request
import sklearn
import time

cap = cv2.VideoCapture(0)



def BMI_Val():
    height = request.form['height']
    weight = request.form['weight']
    height = float(height) / 100
    BMI = float(weight) / (height * height)
    x = round(BMI,2)
    return x

def Class_Value(BMI):
    if (BMI <= 18.5):
        return 'You are UnderWeight'
    elif(18.5 < BMI < 25):
        return 'You are NormalWeight'
    elif (BMI > 25 ):
        return 'You are Overweight'

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return round(angle, 4)

def findDistance(x1, y1, x2, y2):
    dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return dist

with open('knn_all_exe.pkl', 'rb') as f:
    model = pickle.load(f)

def generate_frames(BMI):
    stage = None
    stage2= None
    counter3 =0
    counter2 = 0
    counter1 = 0
    status1 = 'start1'
    status2 ='start2'
    status3 ='start3'
    st2 = 1
    st1 = 1
    st3 = 1

    while True:
        ## read the camera frame
        success,frame=cap.read()
        if not success:
            break
        else:
            mp_pose = mp.solutions.pose

            # Initiate holistic model
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2) as Pose:

                frame_cnt = 0
                while cap.isOpened():

                    ret, frame = cap.read()
                    h, w = frame.shape[:2]

                    # Recolor Feed
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False


                    # Make Detections
                    body_points = Pose.process(image)
                    lm = body_points.pose_landmarks
                    lmPose = mp_pose.PoseLandmark
                    #print(len(lm.landmark))

                    # Recolor image back to BGR for rendering
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    try:


                        l_shldrdis_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
                        l_shldrdis_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
                        r_shldrdis_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
                        r_shldrdis_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)

                        stand = findDistance(l_shldrdis_x, l_shldrdis_y, r_shldrdis_x, r_shldrdis_y)

                        if 82 < stand < 98:

                            cv2.line(image, (450, 100), (450, 500), (0, 255, 0), 4)
                            cv2.line(image, (180, 100), (180, 500), (0, 255, 0), 4)
                        else:

                            cv2.line(image, (180, 100), (180, 500), (0, 0, 255), 4)
                            cv2.line(image, (450, 100), (450, 500), (0, 0, 255), 4)

                        l_shldr = (int(lm.landmark[lmPose.LEFT_SHOULDER].x * w), int(lm.landmark[lmPose.LEFT_SHOULDER].y * h))
                        l_elbow = (int(lm.landmark[lmPose.LEFT_ELBOW].x * w), int(lm.landmark[lmPose.LEFT_ELBOW].y * h))
                        l_wrist = (int(lm.landmark[lmPose.LEFT_WRIST].x * w), int(lm.landmark[lmPose.LEFT_WRIST].y * h))
                        r_shldr = (int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w), int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h))
                        r_elbow = (int(lm.landmark[lmPose.RIGHT_ELBOW].x * w), int(lm.landmark[lmPose.RIGHT_ELBOW].y * h))
                        r_wrist = (int(lm.landmark[lmPose.RIGHT_WRIST].x * w), int(lm.landmark[lmPose.RIGHT_WRIST].y * h))

                        r_hip = (int(lm.landmark[lmPose.RIGHT_HIP].x * w), int(lm.landmark[lmPose.RIGHT_HIP].y * h))
                        r_knee = (int(lm.landmark[lmPose.RIGHT_KNEE].x * w), int(lm.landmark[lmPose.RIGHT_KNEE].y * h))
                        r_ankle = (int(lm.landmark[lmPose.RIGHT_ANKLE].x * w), int(lm.landmark[lmPose.RIGHT_ANKLE].y * h))
                        l_hip = (int(lm.landmark[lmPose.LEFT_HIP].x * w), int(lm.landmark[lmPose.LEFT_HIP].y * h))
                        l_knee = (int(lm.landmark[lmPose.LEFT_KNEE].x * w), int(lm.landmark[lmPose.LEFT_KNEE].y * h))
                        l_ankle = (int(lm.landmark[lmPose.LEFT_ANKLE].x * w), int(lm.landmark[lmPose.LEFT_ANKLE].y * h))
                        nose = (int(lm.landmark[lmPose.NOSE].x * w), int(lm.landmark[lmPose.NOSE].y * h))

                        cv2.circle(image, (l_elbow), 5, (127, 20, 0), 6)
                        cv2.circle(image, (l_wrist), 5, (127, 20, 0), 6)
                        cv2.circle(image, (l_shldr), 5, (127, 20, 0), 6)
                        cv2.circle(image, (r_elbow), 5, (127, 20, 0), 6)
                        cv2.circle(image, (r_wrist), 5, (127, 20, 0), 6)
                        cv2.circle(image, (r_shldr), 5, (127, 20, 0), 6)
                        cv2.circle(image, (l_hip), 5, (127, 20, 0), 6)
                        cv2.circle(image, (l_knee), 5, (127, 20, 0), 6)
                        cv2.circle(image, (l_ankle), 5, (127, 20, 0), 6)
                        cv2.circle(image, (r_ankle), 5, (127, 20, 0), 6)
                        cv2.circle(image, (r_hip), 5, (127, 20, 0), 6)
                        cv2.circle(image, (r_knee), 5, (127, 20, 0), 6)
                        cv2.circle(image, (nose), 5, (127, 20, 0), 6)

                        cv2.line(image, (l_shldr), (l_elbow), (0, 255, 255), 4)
                        cv2.line(image, (r_shldr), (r_elbow), (0, 255, 255), 4)
                        cv2.line(image, (l_elbow), (l_wrist), (0, 255, 255), 4)
                        cv2.line(image, (r_elbow), (r_wrist), (0, 255, 255), 4)
                        cv2.line(image, (l_hip), (l_knee), (0, 255, 255), 4)
                        cv2.line(image, (l_knee), (l_ankle), (0, 255, 255), 4)
                        cv2.line(image, (r_hip), (r_knee), (0, 255, 255), 4)
                        cv2.line(image, (r_knee), (r_ankle), (0, 255, 255), 4)
                        cv2.line(image, (l_shldr), (l_hip), (0, 255, 255), 4)
                        cv2.line(image, (r_shldr), (r_hip), (0, 255, 255), 4)
                        cv2.line(image, (r_hip), (l_hip), (0, 255, 255), 4)
                        cv2.line(image, (r_shldr), (l_shldr), (0, 255, 255), 4)

                        select = (int(lm.landmark[lmPose.LEFT_SHOULDER].x * w), int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)), (
                            int(lm.landmark[lmPose.LEFT_ELBOW].x * w), int(lm.landmark[lmPose.LEFT_ELBOW].y * h)), \
                                (int(lm.landmark[lmPose.LEFT_WRIST].x * w), int(lm.landmark[lmPose.LEFT_WRIST].y * h)), (
                                    int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w), int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)), \
                                (int(lm.landmark[lmPose.RIGHT_ELBOW].x * w), int(lm.landmark[lmPose.RIGHT_ELBOW].y * h)), (
                                    int(lm.landmark[lmPose.RIGHT_WRIST].x * w), int(lm.landmark[lmPose.RIGHT_WRIST].y * h)), \
                                (int(lm.landmark[lmPose.RIGHT_HIP].x * w), int(lm.landmark[lmPose.RIGHT_HIP].y * h)), \
                                (int(lm.landmark[lmPose.RIGHT_KNEE].x * w), int(lm.landmark[lmPose.RIGHT_KNEE].y * h)), \
                                (int(lm.landmark[lmPose.RIGHT_ANKLE].x * w), int(lm.landmark[lmPose.RIGHT_ANKLE].y * h)), \
                                (int(lm.landmark[lmPose.LEFT_HIP].x * w), int(lm.landmark[lmPose.LEFT_HIP].y * h)), \
                                (int(lm.landmark[lmPose.LEFT_KNEE].x * w), int(lm.landmark[lmPose.LEFT_KNEE].y * h)), \
                                (int(lm.landmark[lmPose.LEFT_ANKLE].x * w), int(lm.landmark[lmPose.LEFT_ANKLE].y * h)), \
                                (int(lm.landmark[lmPose.NOSE].x * w), int(lm.landmark[lmPose.NOSE].y * h))
                        tupel_list = list(select)
                        # print(tupel_list)
                        lndmrk = []

                        for t in tupel_list:
                            for x in t:
                                lndmrk.append(x)
                        print('check',lndmrk)

                        # Extract Pose landmarks
                        keypoints = lndmrk

                        # row = pose

                        # Make Detections
                        X = pd.DataFrame([keypoints])
                        print('x', X)

                        body_language_class = model.predict(X)[0]
                        #body_language_prob = model.predict_proba(X)[0]
                        #cv2.putText(image, str(body_language_class), (165, 25), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 2, )
                        if (float(BMI) <= 18.5):
                            print('underweight')


                            if body_language_class == 'Push_up':
                                stage2 = 'now push(Go) down'
                            if body_language_class == 'Push_down' and stage2 == 'now push(Go) down' and status3 == 'start3':
                                counter3 +=  1
                                stage2 = 'now push(Go) up'
                                if counter3 == 10 and st3 == 1:
                                    stage2 = 'pushup set' + ' ' + str(st3) + ' ' + 'complited'
                                    st3 = 2
                                    counter3 = 0
                                if counter3 == 7 and st3 == 2:
                                    stage2 = 'pushup set' + ' ' + str(st3) + ' ' + 'complited'
                                    counter3 = 0
                                    st3 = 3
                                if counter3 == 3 and st3 == 3:
                                    stage2 = 'pushup  set' + ' ' + str(st3) + ' ' + 'complited'
                                    counter3 = 0
                                    status3 = 'stop3'
                                    st3 = 0

                            if body_language_class == 'Rest':
                                stage = 'down'
                            if body_language_class == 'Left_Bicep' and stage == 'down' and status1 == 'start1':
                                counter1 += 1
                                stage = 'left Bicep up_' \
                                        'now go down'
                                if counter1 == 10 and st1 == 1:
                                    stage = 'left Bicep  set' + ' ' + str(st1) + ' ' + 'complited'
                                    st1 = 2
                                    counter1 = 0
                                if counter1 == 7 and st1 == 2:
                                    stage = 'left Bicep  set' + ' ' + str(st1) + ' ' + 'complited'
                                    counter1 = 0
                                    st1 = 3
                                if counter1 == 5 and st1 == 3:
                                    stage = 'left Bicep  set' + ' ' + str(st1) + ' ' + 'complited'
                                    counter1 = 0
                                    status1 = 'Stop1'
                                    st1 = 0
                            if body_language_class == 'Right_Bicep' and stage == 'down' and status2 == 'start2':
                                counter2 += 1
                                stage = 'Right Bicep up' \
                                        'now go down'
                                if counter2 == 10 and st2 == 1:
                                    stage = 'Right Bicep  set' + ' ' + str(st2) + ' ' + 'complited'
                                    st2 = 2
                                    counter2 = 0
                                if counter2 == 7 and st2 == 2:
                                    stage = 'Right Bicep  set' + ' ' + str(st2) + ' ' + 'complited'
                                    counter2 = 0
                                    st2 = 3
                                if counter2 == 5 and st2 == 3:
                                    stage = 'Right Bicep  set' + ' ' + str(st2) + ' ' + 'complited'
                                    counter2 = 0
                                    status2 = 'stop2'
                                    st2 = 0


                            if body_language_class == 'Left_Bicep' or body_language_class == 'Right_Bicep' or \
                                    body_language_class == 'Rest' or  body_language_class =='Push_up' or body_language_class== 'Push_down':
                                cv2.putText(image, str(body_language_class), (165, 25), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 2,)
                            # cv2.putText(image, 'REP-', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
                            cv2.putText(image, str(counter2), (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                            cv2.putText(image, str(counter1), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                            cv2.putText(image, str(counter3), (70, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                            cv2.putText(image, stage, (10, 90), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                            cv2.putText(image, stage2, (10, 90), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                            if frame_cnt < 100:
                                cv2.putText(image, 'Do Bicep and Pushups Exercise for weight gain', (10, 205), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1,
                                        cv2.LINE_AA)
                                cv2.putText(image, 'for weight gain', (10, 235), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1,
                                        cv2.LINE_AA)


                        if (18.5 <float(BMI) < 25):
                            print('Normal Weight')

                            if body_language_class == 'Rest':
                                stage = 'down'
                            if body_language_class == 'Left_Shoulder' and stage == 'down' and status1 == 'start1':
                                counter1 += 1
                                stage = 'left shoulder up_now go down'
                                if counter1 == 10 and st1 == 1:
                                    stage = 'left shoulder  set' + ' ' + str(st1) + ' ' + 'complited'
                                    st1 = 2
                                    counter1 = 0
                                    status1 = 'stop1'
                            if body_language_class == 'Right_Shoulder' and stage == 'down' and status2 == 'start2':
                                counter2 += 1
                                stage = 'Right shoulder up_now go down'
                                if counter2 == 10 and st2 == 1:
                                    stage = 'Right shoulder  set' + ' ' + str(st2) + ' ' + 'complited'
                                    st2 = 2
                                    counter2 = 0
                                    status2 ='stop2'
                            squats_l = calculate_angle(l_hip, l_knee, l_ankle)
                            squats_r = calculate_angle(r_hip, r_knee, r_ankle)

                            if squats_l < 85 and squats_r < 85:
                                stage2 = 'Squats--go up'
                            if squats_l > 169 and squats_r > 169 and stage2 == 'Squats--go up' and status3 == 'start3':
                                counter3 += 1
                                stage2 = ' Squats -- go down'
                                if counter3 == 10 and st3 == 1:
                                    stage2 = 'Squats  set' + ' ' + str(st3) + ' ' + 'complited'
                                    st3 = 2
                                    counter3 = 0
                                if counter3 == 5 and st3 == 2:
                                    stage2 = 'Squats set' + ' ' + str(st3) + ' ' + 'complited'
                                    counter3 = 0
                                    status3 = 'stop3'
                                    st3 = 0

                            if body_language_class == 'Right_Shoulder' or body_language_class == 'Left_Shoulder'or body_language_class == 'Rest':
                                cv2.putText(image, str(body_language_class), (165, 25), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 2,)
                            # cv2.putText(image, 'REP-', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
                            cv2.putText(image, str(counter2), (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                            cv2.putText(image, str(counter3), (70, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                            cv2.putText(image, stage, (10, 90), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                            cv2.putText(image, str(counter1), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                            cv2.putText(image, stage2, (10, 120), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                            if frame_cnt < 100:
                                cv2.putText(image, 'Do Shoulder and squats Exercise',(10, 205), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1,
                                            cv2.LINE_AA)
                                cv2.putText(image, ' to stay Healthy', (10, 235), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1,
                                        cv2.LINE_AA)



                        if (float(BMI) > 25 ):
                            print('OverWeight')
                            if body_language_class == 'Push_up':
                                stage = 'now push(Go) down'
                            if body_language_class == 'Push_down' and stage == 'now push(Go) down' and status1 == 'start1':
                                counter1 +=  1
                                stage = 'now push(Go) up'
                                if counter1 == 10 and st1 == 1:
                                    stage = 'pushup set' + ' ' + str(st1) + ' ' + 'complited'
                                    st1 = 2
                                    counter1 = 0
                                if counter1 == 7 and st1 == 2:
                                    stage = 'pushup set' + ' ' + str(st1) + ' ' + 'complited'
                                    counter1 = 0
                                    st3 = 3
                                if counter1 == 5 and st1 == 3:
                                    stage = 'pushup  set' + ' ' + str(st1) + ' ' + 'complited'
                                    counter1 = 0
                                    status1 = 'stop1'
                                    st1 = 0



                            squats_l = calculate_angle(l_hip, l_knee, l_ankle)
                            squats_r = calculate_angle(r_hip, r_knee, r_ankle)

                            if squats_l < 85 and squats_r < 85:
                                stage2 = 'Squats--go up'
                            if squats_l >169 and squats_r >169 and stage2 == 'Squats--go up' and status2 == 'start2':
                                counter2 += 1
                                stage2 = ' Squats -- go down'
                                if counter2 == 10 and st2 == 1:
                                    stage2 = 'Squats  set' + ' ' + str(st2) + ' ' + 'complited'
                                    st2 = 2
                                    counter2 = 0
                                if counter2 == 7 and st2 == 2:
                                    stage2 = 'Squats  set' + ' ' + str(st2) + ' ' + 'complited'
                                    st2 = 3
                                    counter2 = 0
                                if counter2 == 5 and st2 == 3:
                                    stage2 = 'Squats set' + ' ' + str(st2) + ' ' + 'complited'
                                    counter2 = 0
                                    status2 = 'start2'
                                    st2 = 0




                            if body_language_class == 'Push_up' or body_language_class == 'Push_down' or body_language_class == 'Rest':
                                cv2.putText(image, str(body_language_class), (165, 25), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 2, )

                            #cv2.putText(image, 'REP-', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
                            cv2.putText(image, str(counter2), (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                            cv2.putText(image, str(counter1), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                            cv2.putText(image, stage, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                            cv2.putText(image, stage2, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                            if frame_cnt < 100:
                                cv2.putText(image, 'Do Pushups and squats Exercise',(10, 205), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1,
                                            cv2.LINE_AA)
                                cv2.putText(image, ' For weight loss', (10, 235), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1,
                                        cv2.LINE_AA)




                        #cv2.putText(image, 'Exercise-', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)


                        # Display Probability
                        # cv2.putText(image, 'PROB'
                        #  ,(15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        # cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)], 2))
                        #  ,(10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                        frame_cnt += 1
                    except:
                        pass
                    ret,buffer=cv2.imencode('.jpg',image)
                    frame=buffer.tobytes()
                    yield(b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

                    # cv2.imshow('AI fitness', image)

                    # if cv2.waitKey(10) & 0xFF == ord('q'):
                    #     break

            cap.release()
            cv2.destroyAllWindows()
