import cv2
import mediapipe as mp
import time
from twilio.rest import Client

account_sid = 'blah blah'
auth_token = 'blah blah'
client = Client(account_sid, auth_token)

# Initialize the video capture and face mesh
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Time variables for detecting prolonged eye closure
eyes_closed_start_time = None
eyes_closed_duration_threshold = 3  # seconds
notification_sent = False

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmarks_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmarks_points:
        landmarks = landmarks_points[0].landmark
        left_eye = [landmarks[145], landmarks[159]]
        right_eye = [landmarks[374], landmarks[386]]

        for landmark in left_eye:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (255, 255, 0), -1)

        for landmark in right_eye:
            r = int(landmark.x * frame_w)
            f = int(landmark.y * frame_h)
            cv2.circle(frame, (r, f), 3, (0, 255, 0), -1)

        left_eye_aspect_ratio = abs(left_eye[0].y - left_eye[1].y)
        right_eye_aspect_ratio = abs(right_eye[0].y - right_eye[1].y)

        if left_eye_aspect_ratio < 0.004 and right_eye_aspect_ratio < 0.004:
            if eyes_closed_start_time is None:
                eyes_closed_start_time = time.time()
            elif time.time() - eyes_closed_start_time > eyes_closed_duration_threshold:
                if not notification_sent:
                    print("Wake up!")
                    notification_sent = True
                    message = client.messages.create(
                        from_='+18446093286',
                        body='Wake up!',
                        to='+12244192194'
                    )
                    print("Message sent")
        else:
            eyes_closed_start_time = None
            notification_sent = False

    cv2.imshow('Sleep Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.01)

cam.release()
cv2.destroyAllWindows()


