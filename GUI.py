import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
import os
from keras.models import load_model
from keras.preprocessing import image


def setResetBackground():
    global bgModel
    bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
    
def removeBG(frame):
    fgMask = bgModel.apply(frame,learningRate=learningRate)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
    fgMask = cv2.erode(fgMask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgMask)
    return res

def predict(path):
    preds = model_predict(path, model)
    preds = np.argmax(preds, axis = -1)
    result = str(preds)
    print(result)
    if result == "[0]": result="INDEX"
    elif result == "[1]": result="PEACE"
    elif result == "[2]": result="THREE"
    elif result == "[3]": result="PALM OPENED"
    elif result == "[4]": result="PALM CLOSED"
    elif result == "[5]": result="OK"
    elif result == "[6]": result="THUMBS"
    elif result == "[7]": result="FIST"
    elif result == "[8]": result="SWING"
    elif result == "[9]": result="SMILE"
    else: result="CAN NOT BE IDENTIFIED"
    resultText = "Detected Gesture is: " + result
    detectionLabel.config(text=resultText)

def model_predict(img_path, model):
    img = image.load_img(img_path, color_mode = 'grayscale', target_size=(50, 50))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    return preds


def showVid():
    global imageCounter
    global temp
    global binary
    ret, frame = camera.read()
    #color = frame
    frame = cv2.bilateralFilter(frame, 5, 50, 100)
    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (int(capRegionXBegin * frame.shape[1]), 0),
                 (frame.shape[1], int(capRegionYEnd * frame.shape[0])), (255, 0, 0), 2)
    
    #if isBgCaptured == 1:
    color = frame[0:int(capRegionYEnd * frame.shape[0]), int(capRegionXBegin * frame.shape[1]):frame.shape[1]]
    img = removeBG(frame)
    mask = img[0:int(capRegionYEnd * frame.shape[0]), int(capRegionXBegin * frame.shape[1]):frame.shape[1]]
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    #gray = cv2.equalizeHist(gray)
    gray = cv2.resize(gray, (320, 240))
    
    #blur = cv2.GaussianBlur(gray, (5, 5), 0)
    blur = cv2.medianBlur(gray,5)
    
    img = Image.fromarray(gray)
    imgtk = ImageTk.PhotoImage(image = img)
    grayImage.imgtk = imgtk
    grayImage.configure(image = imgtk)
    
    img = Image.fromarray(blur)
    imgtk = ImageTk.PhotoImage(image = img)
    binaryImage.imgtk = imgtk
    binaryImage.configure(image = imgtk)
        
    if imageCounter < 100 and dataCollectionMode == True:
        imageCounter += 1
        imageNumber = "{:03d}".format(imageCounter)
        imageName = "train_person_07_" + imageNumber + ".png"
        grayImgPath = 'Data/Gray/Training_Set/' + folders[8]
        cv2.imwrite(os.path.join(grayImgPath , imageName), gray)
        oriImagePath = 'Data/Original/Training_Set/' + folders[8]
        cv2.imwrite(os.path.join(oriImagePath, imageName), color)
        blurImagePath = 'Data/ReducedNoise/Training_Set/' + folders[8]
        cv2.imwrite(os.path.join(blurImagePath, imageName), blur)
        print("{} written!".format(imageName))
            
    if predictionMode == True:
        path = 'RealTimeImage'
        cv2.imwrite(os.path.join(path, 'detection.png'), gray)
        path = 'RealTimeImage/detection.png'
        predict(path)
        
        
    
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    capturedImage.imgtk = imgtk
    capturedImage.configure(image=imgtk)
    capturedImage.after(200, showVid)
    
def dataModeOn():
    global dataCollectionMode
    dataCollectionMode = True
    setBackButton.grid_remove()
    startDataButton.grid_remove()
    stopDataButton.grid(row=3, column =1)
    startlLivePredButton.grid_remove()
    
def dataModeOff():
    global dataCollectionMode
    dataCollectionMode = False
    global imageCounter
    imageCounter = 0
    setBackButton.grid(row = 3, column = 0)
    stopDataButton.grid_remove()
    startDataButton.grid(row = 3, column = 1)
    startlLivePredButton.grid(row = 3, column = 2)

def predModeOn():
    global predictionMode
    predictionMode = True
    startDataButton.grid_remove()
    startlLivePredButton.grid_remove()
    stopLivePredutton.grid(row = 3, column = 2)
    
def predModeOff():
    global predictionMode
    predictionMode = False
    stopLivePredutton.grid_remove()
    startDataButton.grid(row = 3, column = 1)
    startlLivePredButton.grid(row = 3, column = 2)
    detectionLabel.config(text="")
 

capRegionXBegin = 0.5
capRegionYEnd = 0.8
threshold = 60
blurValue = 41
bgSubThreshold = 50
learningRate = 0

imageCounter = 0
dataCollectionMode = False
predictionMode = False
bgModel = None
temp = None

folders = ["01_INDEX", "02_PEACE", "03_THREE", "04_PALM_OPENED", "05_PALM_CLOSED", "06_OK", "07_THUMBS", "08_FIST", "09_SWING", "10_SMILE"]

camera = cv2.VideoCapture(0)
ret = camera.set(3, 640)
ret = camera.set(4, 480)

MODEL_PATH = 'models/model_10.h5'
model = load_model(MODEL_PATH)
model._make_predict_function()
print('Model loaded. Start serving...')

bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)

root=tk.Tk()
root.title("Hand Gesture Detection")
root.configure(background = "white")  
tk.Label(root, text = "Live Gesture Detection", bg = "white", fg = "black", font = "none 12 bold").grid(row = 0, column = 0)    
capturedImage = tk.Label(root)
capturedImage.grid(row = 1, column = 0)
preprocessedImageContainer = tk.Frame(root)
preprocessedImageContainer.grid(row = 1, column = 1)
grayImage = tk.Label(preprocessedImageContainer)
grayImage.grid(row = 1, column = 1)
binaryImage = tk.Label(preprocessedImageContainer)
binaryImage.grid(row = 2, column = 1)
buttonContainer = tk.Frame(root)
buttonContainer.grid(row = 3, column = 0)
setBackButton = tk.Button(buttonContainer, text = "Reset Background", command = setResetBackground)
setBackButton.grid(row = 3, column = 0)    
startDataButton = tk.Button(buttonContainer, text = "Start Data Collection", command = dataModeOn)
startDataButton.grid(row = 3, column = 1)
stopDataButton = tk.Button(buttonContainer, text = "Stop Data Collection", command = dataModeOff)
startlLivePredButton = tk.Button(buttonContainer, text = "Start Detection", command = predModeOn)
startlLivePredButton.grid(row = 3, column = 2)
stopLivePredutton = tk.Button(buttonContainer, text = "Stop Detection", command = predModeOff)    
detectionLabel = tk.Label(root, text="", bg="white", fg="black", font="none 12 bold")
detectionLabel.grid(row = 3, column = 1)


showVid()
  
root.mainloop()
camera.release()