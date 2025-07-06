#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().system('pip install mediapipe opencv-python')


# In[3]:


# get_ipython().system('pip install jupyter streamlit')


# In[ ]:


import  cv2 as cv
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# In[ ]:


# capturing video
cap = cv.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    cv.imshow("Pose and Exercise Detection AI", frame)
    
    if cv.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()


# In[ ]:


def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
    return angle


# In[ ]:


cap = cv.VideoCapture(0)

counter = 0;
stage = None

with mp_pose.Pose( min_detection_confidence =0.5 , min_tracking_confidence = 0.5) as pose:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        image = cv.cvtColor(frame ,cv.COLOR_BGR2RGB)
        
        results = pose.process(image)
        
        image = cv.cvtColor(image ,cv.COLOR_RGB2BGR)
        
        #Extracting landmarks

        try:
            landmarks = results.pose_landmarks.landmark
        
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        
            angle = calculate_angle(shoulder,elbow, wrist)
        
            cv.putText(image,str(angle),
                  tuple(np.multiply(elbow, [640,480]).astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),2, cv.LINE_AA)

            if angle >160:
                stage = "down"
            if angle <30 and stage == "down":
                stage = "up"
                counter+=1
                print(counter)

        except:
            pass

        cv.rectangle(image, (0,0), (225,73), (245,117,16), -1)

        cv.putText(image, 'CNT', (15,12),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)

        cv.putText(image, str(counter),
                  (10,60),
                  cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv.LINE_AA)

        cv.putText(image, 'STAGE', (65,12),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)

        cv.putText(image, stage,
                  (60,60),
                  cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv.LINE_AA)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=5),
                                 mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3, circle_radius=5))
        cv.imshow("Pose and Exercise Detection AI", image)
        
        if cv.waitKey(10) & 0xFF == ord('q'):
            break


cap.release()
cv.destroyAllWindows()


# In[ ]:


import cv2 as cv
import numpy as np
import mediapipe as mp


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    return angle


def update_reps_and_curls_and_squats(left_angle, right_angle, left_knee_angle, right_knee_angle, left_hip_angle, right_hip_angle, 
                                     stage, counter, curl_counter, curl_stage, squat_counter, squat_stage, exercise_type):
    
    if exercise_type == 'press':
        if left_angle > 160 and right_angle > 160:  
            stage = "down"
        if left_angle < 30 and right_angle < 30 and stage == "down":
            stage = "up"
            counter += 1

    
    if exercise_type == 'curl':
        if left_angle < 30:  
            curl_stage = "up"
        if left_angle > 140 and curl_stage == "up":  
            curl_stage = "down"
            curl_counter += 1

    
    thr = 165  
    if exercise_type == 'squat':
        
        if (left_knee_angle < thr) and (right_knee_angle < thr) and (left_hip_angle < thr) and (right_hip_angle < thr):
            squat_stage = "down"
        
        
        if (left_knee_angle > thr) and (right_knee_angle > thr) and (left_hip_angle > thr) and (right_hip_angle > thr) and (squat_stage == 'down'):
            squat_stage = 'up'
            squat_counter += 1
    
    return stage, counter, curl_counter, curl_stage, squat_counter, squat_stage


cap = cv.VideoCapture(0)
counter = 0  
curl_counter = 0  
squat_counter = 0  
stage = None
curl_stage = None
squat_stage = None
exercise_type = None  


def display_exercise_options(image):
    cv.putText(image, 'Select Exercise: ', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(image, 'Curls [1]  Presses [2]  Squats [3]', (50, 90), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv.LINE_AA)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        
        if exercise_type is None:
            display_exercise_options(image)
        else:
            try:
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                
                    
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                    
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                    
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    
                    
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                    
                    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

                    
                    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

                   
                    left_hip_angle = calculate_angle(left_knee, left_hip, left_ankle)
                    right_hip_angle = calculate_angle(right_knee, right_hip, right_ankle)

                    
                    cv.putText(image, f'Left Arm: {left_angle:.2f}', tuple(np.multiply(left_elbow, [640, 480]).astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
                    cv.putText(image, f'Right Arm: {right_angle:.2f}', tuple(np.multiply(right_elbow, [640, 480]).astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
                    cv.putText(image, f'Left Knee: {left_knee_angle:.2f}', tuple(np.multiply(left_knee, [640, 480]).astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
                    cv.putText(image, f'Right Knee: {right_knee_angle:.2f}', tuple(np.multiply(right_knee, [640, 480]).astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
                    cv.putText(image, f'Left Hip: {left_hip_angle:.2f}', tuple(np.multiply(left_hip, [640, 480]).astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
                    cv.putText(image, f'Right Hip: {right_hip_angle:.2f}', tuple(np.multiply(right_hip, [640, 480]).astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)

                    
                    stage, counter, curl_counter, curl_stage, squat_counter, squat_stage = update_reps_and_curls_and_squats(
                        left_angle, right_angle, left_knee_angle, right_knee_angle, left_hip_angle, right_hip_angle,
                        stage, counter, curl_counter, curl_stage, squat_counter, squat_stage, exercise_type)
            
            except Exception as e:
                print(f"Error occurred: {e}")

            
            cv.rectangle(image, (0, 0), (225, 160), (245, 117, 16), -1)

            if exercise_type == 'press':
                cv.putText(image, 'PRESSES', (65, 12), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
                cv.putText(image, str(counter), (60, 60), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv.LINE_AA)

            elif exercise_type == 'curl':
                cv.putText(image, 'CURLS', (65, 12), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
                cv.putText(image, str(curl_counter), (60, 60), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv.LINE_AA)

            elif exercise_type == 'squat':
                cv.putText(image, 'SQUATS', (65, 12), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
                cv.putText(image, str(squat_counter), (60, 60), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv.LINE_AA)

            
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                     mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=5),
                                     mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3, circle_radius=5))

        cv.imshow("Pose and Exercise Detection AI", image)

       
        key = cv.waitKey(1) & 0xFF
        if key == ord('1') and exercise_type is None:  
            exercise_type = 'curl'
        elif key == ord('2') and exercise_type is None:  
            exercise_type = 'press'
        elif key == ord('3') and exercise_type is None:  
            exercise_type = 'squat'

        if key == ord('q'):  
            break

cap.release()
cv.destroyAllWindows()


# In[2]:


#final code

import cv2 as cv
import numpy as np
import mediapipe as mp
import pyttsx3

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    return angle

def update_reps_and_curls_and_squats(left_angle, right_angle, left_knee_angle, right_knee_angle, left_hip_angle, right_hip_angle, 
                                     stage, counter, curl_counter, curl_stage, squat_counter, squat_stage, exercise_type):
    
    if exercise_type == 'press':
        if left_angle > 160 and right_angle > 160:  
            stage = "down"
        if left_angle < 30 and right_angle < 30 and stage == "down":
            stage = "up"
            counter += 1

    
    if exercise_type == 'curl':
        if left_angle < 30:  
            curl_stage = "up"
        if left_angle > 140 and curl_stage == "up":  
            curl_stage = "down"
            curl_counter += 1

    
    thr = 180
    if exercise_type == 'squat':
        
        if (left_knee_angle < thr) and (right_knee_angle < thr) and (left_hip_angle < thr) and (right_hip_angle < thr):
            squat_stage = "down"
        
        
        if (left_knee_angle > thr) and (right_knee_angle > thr) and (left_hip_angle > thr) and (right_hip_angle > thr) and (squat_stage == 'down'):
            squat_stage = 'up'
            squat_counter += 1
    
    return stage, counter, curl_counter, curl_stage, squat_counter, squat_stage


def calculate_bmi(weight, height):
    bmi = weight / (height ** 2)  
    return bmi

def get_diet_plan(bmi):
    if bmi < 18.5:
        diet_plan = "You are underweight. A high-calorie diet with healthy fats and proteins is recommended. Include more whole grains, nuts, and avocados."
    elif 18.5 <= bmi < 24.9:
        diet_plan = "You have a normal weight. Maintain a balanced diet with lean proteins, vegetables, fruits, and whole grains."
    elif 25 <= bmi < 29.9:
        diet_plan = "You are overweight. Consider reducing your calorie intake and focusing on a balanced diet with low fats and sugars."
    else:
        diet_plan = "You are obese. A weight loss plan with a calorie deficit, increased physical activity, and a nutrient-rich diet is recommended. Consult a healthcare provider for personalized advice."
    
    return diet_plan

def speak_text(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  
    engine.setProperty('volume', 1)  
    try:
        engine.setProperty('pitch', 50)  
    except:
        print("Pitch adjustment is not supported with the SAPI5 engine.")
    engine.say(text)
    engine.runAndWait()

cap = cv.VideoCapture(0)
counter = 0  
curl_counter = 0  
squat_counter = 0  
stage = None
curl_stage = None
squat_stage = None
exercise_type = None  

def display_exercise_options(image):
    cv.putText(image, 'Select Exercise: ', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(image, 'Curls [1]  Presses [2]  Squats [3]', (50, 90), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv.LINE_AA)

def get_user_bmi_input():
    print("Please enter your weight in kilograms: ")
    weight = float(input())
    print("Please enter your height in meters: ")
    height = float(input())
    bmi = calculate_bmi(weight, height)
    diet_plan = get_diet_plan(bmi)
    
    print(f"Your BMI is {bmi:.2f}. {diet_plan}")
    
    speak_text(f"Your BMI is {bmi:.2f}. {diet_plan}")
    
    return bmi, diet_plan

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    bmi, diet_plan = get_user_bmi_input()

    while cap.isOpened():
        ret, frame = cap.read()
        
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        if exercise_type is None:
            display_exercise_options(image)
        else:
            try:
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                
                    
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                    
                    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

                    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

                    left_hip_angle = calculate_angle(left_knee, left_hip, left_ankle)
                    right_hip_angle = calculate_angle(right_knee, right_hip, right_ankle)

                
                    cv.putText(image, f'Left Arm: {left_angle:.2f}', tuple(np.multiply(left_elbow, [640, 480]).astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
                    cv.putText(image, f'Right Arm: {right_angle:.2f}', tuple(np.multiply(right_elbow, [640, 480]).astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
                    cv.putText(image, f'Left Knee: {left_knee_angle:.2f}', tuple(np.multiply(left_knee, [640, 480]).astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
                    cv.putText(image, f'Right Knee: {right_knee_angle:.2f}', tuple(np.multiply(right_knee, [640, 480]).astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
                    cv.putText(image, f'Left Hip: {left_hip_angle:.2f}', tuple(np.multiply(left_hip, [640, 480]).astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
                    cv.putText(image, f'Right Hip: {right_hip_angle:.2f}', tuple(np.multiply(right_hip, [640, 480]).astype(int)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)

                
                    stage, counter, curl_counter, curl_stage, squat_counter, squat_stage = update_reps_and_curls_and_squats(
                        left_angle, right_angle, left_knee_angle, right_knee_angle, left_hip_angle, right_hip_angle,
                        stage, counter, curl_counter, curl_stage, squat_counter, squat_stage, exercise_type)
            
            except Exception as e:
                print(f"Error occurred: {e}")

            cv.rectangle(image, (0, 0), (225, 160), (245, 117, 16), -1)

            if exercise_type == 'press':
                cv.putText(image, 'PRESSES', (65, 12), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
                cv.putText(image, str(counter), (60, 60), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv.LINE_AA)

            elif exercise_type == 'curl':
                cv.putText(image, 'CURLS', (65, 12), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
                cv.putText(image, str(curl_counter), (60, 60), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv.LINE_AA)

            elif exercise_type == 'squat':
                cv.putText(image, 'SQUATS', (65, 12), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
                cv.putText(image, str(squat_counter), (60, 60), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv.LINE_AA)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                     mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=5),
                                     mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3, circle_radius=5))

        
        cv.imshow("Pose and Exercise Detection AI", image)

        key = cv.waitKey(1) & 0xFF
        if key == ord('1') and exercise_type is None:  
            exercise_type = 'curl'
        elif key == ord('2') and exercise_type is None:  
            exercise_type = 'press'
        elif key == ord('3') and exercise_type is None:  
            exercise_type = 'squat'

    
        if key == ord('q'):  
            break

cap.release()
cv.destroyAllWindows()


# In[ ]:




