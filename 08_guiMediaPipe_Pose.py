from tkinter import *
from PIL import Image
from PIL import ImageTk
import cv2
import imutils
import customtkinter
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

# https://omes-va.com/tkinter-opencv-video/

def iniciar():
    global cap
    cap = cv2.VideoCapture("Danza.mp4")  #, cv2.CAP_DSHOW
    visualizar()


def visualizar():
    global cap
    global det
    if cap is not None:
        ret, frame = cap.read()
        if ret == True:
            
            if det :
                frame = modificar(frame) #cv2.bilateralFilter(frame, 15, 100, 100)

            frame = imutils.resize(frame, width=640)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            im = Image.fromarray(frame)
            
            img = customtkinter.CTkImage(light_image=im, size=( frame.shape[1], frame.shape[0]))            

            lblVideo.configure(image=img)
            lblVideo.image = img
            lblVideo.after(10, visualizar)
        else:
            lblVideo.image = ""
            cap.release()

def detectar():
    global det
    det = not det 

def finalizar():
    global cap
    global det
    det = False
    cap.release()

def modificar (fram):
    #fram = cv2.GaussianBlur(fram,(25,25),0)
    proc_frame = detectorPose(fram)
    return proc_frame

# Utilidades Media Pipe------------------------------------

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image



def detectorPose (img, mask = False):
  global detector

  # STEP 3: Load the input image.
  image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)#mp.Image.create_from_file("image.jpg")

  # STEP 4: Detect pose landmarks from the input image.
  detection_result = detector.detect(image)

  if mask:
    mascara(detection_result)

  # STEP 5: Process the detection result. In this case, visualize it.
  annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
  return annotated_image

def mascara(detection_result):
  segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
  visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
  cv2.imshow("Mascara", visualized_mask)
  cv2.waitKey(0)
  # Para destruir todas las ventanas creadas
  cv2.destroyAllWindows()



cap = None
det = False

# STEP 2: Create an PoseLandmarker object.

model_file = open('pose_landmarker.task', "rb")
model_data = model_file.read()
model_file.close()

base_options = python.BaseOptions(model_asset_buffer=model_data)
options = vision.PoseLandmarkerOptions(
  base_options=base_options,
  output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)



root = customtkinter.CTk()

btnIniciar = customtkinter.CTkButton(root, text="Iniciar", width=45, command=iniciar)
btnIniciar.grid(column=0, row=0, padx=5, pady=5)

btnFinalizar = customtkinter.CTkButton(root, text="Finalizar", width=45, command=finalizar)
btnFinalizar.grid(column=1, row=0, padx=5, pady=5)

btnMediaPipe = customtkinter.CTkButton(root, text="Detectar", width=45, command=detectar)
btnMediaPipe.grid(column=2, row=0, padx=5, pady=5)


lblVideo = customtkinter.CTkLabel(root, text="")
lblVideo.grid(column=0, row=1, columnspan=3)

root.resizable(width= False, height  =False)

root.mainloop()