import cv2
import pickle
import mediapipe as mp
import serial
import time
from collections import deque
import pyttsx3
import threading
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

model_file = r'E:\Projects\UG\Atharv _Deshpande\HandSign_n2\model.p' # Update to the correct model file path

import os
if not os.path.exists(model_file):
    print(f"Error: Model file {model_file} not found.")
    exit()

with open(model_file, 'rb') as f:
    model = pickle.load(f)['model']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       min_detection_confidence=0.75,
                       min_tracking_confidence=0.75)

labels_dict = {0: 'D', 1: 'W', 2: 'B'}

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

arduino_port = 'COM5'  # Update this to the correct COM port
baud_rate = 9600

try:
    ser = serial.Serial(arduino_port, baud_rate)
    time.sleep(2)
except serial.SerialException:
    ser = None

predictions_queue = deque(maxlen=5)

frame_counter = 0
process_interval = 5
last_spoken_command = ""

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        if frame_counter % process_interval != 0:
            continue

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        command = ''
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_list = [lm.x for lm in hand_landmarks.landmark] + \
                                [lm.y for lm in hand_landmarks.landmark]

                if len(landmark_list) == 42:
                    try:
                        prediction = model.predict([landmark_list])
                        predicted_letter = prediction[0]

                        predictions_queue.append(predicted_letter)
                        most_frequent_prediction = max(set(predictions_queue), key=predictions_queue.count)

                        if most_frequent_prediction == 'D':
                            command = 'Deposit'
                        elif most_frequent_prediction == 'W':
                            command = 'Withdraw'
                        elif most_frequent_prediction == 'B':
                            command = 'Balance Check'

                        if ser and command:
                            ser.write((command + '\n').encode('utf-8'))

                        if command and command != last_spoken_command:
                            tts_thread = threading.Thread(target=speak_text, args=(command,))
                            tts_thread.start()
                            last_spoken_command = command
                    except Exception as e:
                        pass

        if command:
            cv2.putText(frame, f"Command: {command}",
                        (frame.shape[1] - 250, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Banking Command Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    if ser:
        ser.close()
    cv2.destroyAllWindows()
