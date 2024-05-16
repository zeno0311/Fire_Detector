Project Overview:
This Python script utilizes a Convolutional Neural Network (CNN) model to detect fire in real-time using a camera feed. It includes functionalities for email notifications, sound alerts, and geolocation tracking.
Dependencies:
OpenCV (cv2)
NumPy
Keras
smtplib
email.message
winsound
geocoder
Setup Instructions:
Install the required libraries using pip:
pip install opencv-python numpy keras smtplib geocoder

Update the email credentials in the email_alert function with your own Gmail account details.
Ensure the directory path for the training data in train_generator matches your dataset location.
Running the Script:
Execute the script to start the camera feed and detect fire in real-time.
Adjust the threshold parameter in predict_fire_from_camera to set the sensitivity of fire detection.
