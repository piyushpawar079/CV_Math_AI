import cv2
import numpy as np
import google.generativeai as genai
from cvzone.HandTrackingModule import HandDetector as hd
from PIL import Image
import streamlit as st

col1, col2 = st.columns([2, 1])

with col1:
    run = st.checkbox('Run', value=True)
    FRAME_WINDOW = st.image([])

with col2:
    st.title("Answer")
    output_text_area = st.subheader('')

cam = cv2.VideoCapture(0)
cam.set(3, 1280)
cam.set(4, 720)

detector = hd()

prev_pos = canvas = combined = None
api_key = 'AIzaSyBLoq2qPnvxqfGfyqZb2ifo202nkcPPKKA'
output_text = ''

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

def getHandInfo(img):

    hands, img = detector.findHands(img, flipType=False)

    if hands:
        hand = hands[0]
        lmList = detector.findPosition(img)
        fingers = detector.fingersUp(hand)

        return fingers, lmList
    else:
        return None

def draw(info, prev_pos, canvas):

    fingers, lmList = info
    current_pos = None

    if fingers[1] and not fingers[2]:

        current_pos = lmList[8][1], lmList[8][2]

        if prev_pos is None:
            prev_pos = current_pos

        cv2.line(canvas, current_pos, prev_pos, (255, 0, 255), 10)

    elif fingers == [1, 1, 1, 1, 1]:
        canvas = np.zeros_like(img)

    return current_pos, canvas

def sendToAI(canvas, fingers, model):

    if fingers == [0, 1, 1, 1, 0]:
        pil_img = Image.fromarray(canvas)
        response = model.generate_content(["solve this math problem: ", pil_img])

        return response.text

while True:
    _, img = cam.read()
    img = cv2.flip(img, 1)

    if canvas is None:
        canvas = np.zeros_like(img)

    info = getHandInfo(img)

    if info:
        fingers, lmList = info
        prev_pos, canvas = draw(info, prev_pos, canvas)
        output_text = sendToAI(canvas, fingers, model)

    combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)

    FRAME_WINDOW.image(combined, channels='BGR')
    output_text_area.text(output_text)
    # cv2.imshow("combined", combined)
    cv2.waitKey(1)