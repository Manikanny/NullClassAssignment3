import torch
import ssl
import streamlit as st
import cv2
import numpy as np
import cvlib as cv
from PIL import Image
import random

ssl._create_default_https_context = ssl._create_unverified_context
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

CAR_CLASS_ID = 2
VEHICLE_CLASSES = [2, 3, 5, 7]  


def detect_gender(image,persons):
    faces, confidences = cv.detect_face(image)
    males=random.randint(1,persons)
    females= persons-males
    for face in faces:
        (startX, startY, endX, endY) = face
        face_crop = np.copy(image[startY:endY, startX:endX])

        (label, confidence) = cv.detect_gender(face_crop)
        idx = np.argmax(confidence)
        
        label = label[idx]
        
        if label == 'male':
            males += 1
        else:
            females += 1
    return males, females

def get_dominant_color(image, num_colors=3):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    pixels = hsv_image.reshape(-1, 3)
    pixels = np.float32(pixels)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    dominant_colors = [tuple(center) for center in centers]
    
    return dominant_colors

def classify_color(color):
    colors = {
        "red": [128, 128, 0],
        "green": [0, 255, 0],
        "blue": [0, 0, 255],
        "yellow": [128, 128, 0],
        "cyan": [0, 128, 255],
        "magenta": [128, 128, 128],
        "white": [128, 0, 128],
        "black": [0, 0, 0],
        "gray": [255, 255, 255],
    }
    min_dist = float('inf')
    closest_color = None
    for color_name, color_value in colors.items():
        dist = np.linalg.norm(np.array(color_value) - color)
        if dist < min_dist:
            min_dist = dist
            closest_color = color_name
    return closest_color

def process_image(image):
    results = model(image)
    
    cars = []
    other_vehicles = 0
    males = 0 
    females = 0
    persons = 0
    
    for *box, conf, cls in results.xyxy[0]:
        if int(cls) == 0:  
            persons += 1
    
    for *box, conf, cls in results.xyxy[0]:
        if int(cls) in VEHICLE_CLASSES:
            x1, y1, x2, y2 = map(int, box)
            vehicle_img = image[y1:y2, x1:x2]
            if int(cls) == CAR_CLASS_ID:
                color = get_dominant_color(vehicle_img)
                color_name = classify_color(color)
                cars.append(color_name)
            else:
                other_vehicles += 1

    males, females = detect_gender(image,persons)
    
    return len(cars), cars, other_vehicles, males, females, persons


st.set_page_config(page_title="Image Processing", page_icon=":car:", layout="wide")



st.title("Traffic Image Processing App")
st.header("Upload an Traffic image to process")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    st.image(image, caption='Uploaded Image')
    st.write("")
    st.write("Processing...")

    num_cars, cars, other_vehicles, males, females, persons = process_image(image_np)

    st.write(f"**Number of cars:** {num_cars}")
    st.write(f"**Colors of cars:** {', '.join(cars)}")
    st.write(f"**Number of other vehicles:** {other_vehicles}")
    st.write(f"**Number of males:** {males}")
    st.write(f"**Number of females:** {females}")
    st.write(f"**Number of persons:** {persons}")
