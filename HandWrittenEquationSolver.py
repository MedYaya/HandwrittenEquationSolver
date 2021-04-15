import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt
import os
from tkinter import *
from tkinter import filedialog


     
def browseFiles(): 
    filename = filedialog.askopenfilename(initialdir = "/", title = "Select a File", filetypes = (("Text files", "*.png*"), ("all files", "*.*"))) 
    
    label_file_explorer.configure(text="File Opened: "+filename)

    img = cv2.imread(filename)
    #plt.imshow(img)
    #plt.show()
    
    #input: equation photo
    #output: array of each number/operation photo
    splittedImages = preprocessing(img)
    #input: array of each number/operation photo
    #output: array of the predicted values
    predictedValues = predict(splittedImages)
    #input: array of the predicted values
    #output: solution of the equation
    solution = execute(predictedValues)
    print(solution)

def preprocessing(img):
    image = img
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    ROI_number = 0
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    xs = []

    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        xs.append(x)

    xs.sort()
    print(xs)

    returnedImages = np.array([])
    returnedImages.resize(len(xs),120,120,3)

    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)

        for i in range(0,len(xs)):
            if xs[i] == x:
                break

        print(i)

        if w > h:
            ph = h
            h = w
            y = y - int((h - ph)/2)
        else:
            pw = w
            w = h
            x = x - int((w - pw)/2)

        x = x - 20
        y = y - 20
        w = w + 40
        h = h + 40

        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
        ROI = original[y:y+h, x:x+w]
        cv2.imwrite('SplitOutput/{}.png'.format(i), ROI)
        ROI = cv2.resize(ROI,(120,120))
        returnedImages[i] = ROI
        ROI_number += 1

    cv2.imshow('image', image)
    return returnedImages

def predict(inputImages):
    model = load_model('Model_1.model')

    images = inputImages

    predictions = model.predict(images) 

    predictedValues = []


    for i in range(0,len(images)):
         
        predictedValues.append(np.argmax(predictions[i]))

        if(predictedValues[i] == 10):
            predictedValues[i] = '-'
        elif(predictedValues[i] == 11):
            predictedValues[i] = '+'
        elif(predictedValues[i] == 12):
            predictedValues[i] = '*'

        print('prediction -> ',predictedValues[i])
        label_file_explorer = Label(window, 
							text = ("prediction -> "+str(predictedValues[i])), 
							width = 100, height = 4, 
							fg = "blue")
        label_file_explorer.grid(column = 1, row = 4+i)
        #cv2.imshow('image', images[i])
        #cv2.waitKey()

    return predictedValues

def execute(predictedValues):
    
    equationStatement = ""

    for i in range(0,len(predictedValues)):
        equationStatement += str(predictedValues[i])

    solution = eval(equationStatement)
    label_file_explorer = Label(window, 
							text = (equationStatement+" = "+str(solution)), 
							width = 100, height = 4, 
							fg = "blue")
    label_file_explorer.grid(column = 1, row = 5+len(predictedValues))
    return solution

#Building the GUI
window = Tk()
window.title('File Explorer') 
window.geometry("700x400") 
window.config(background = "white") 
label_file_explorer = Label(window, 
							text = "File Explorer using Tkinter", 
							width = 100, height = 4, 
							fg = "blue") 

button_explore = Button(window, 
						text = "Browse Files", 
						command = browseFiles) 
button_exit = Button(window, 
					text = "Exit", 
					command = exit) 
label_file_explorer.grid(column = 1, row = 1) 
button_explore.grid(column = 1, row = 2) 
button_exit.grid(column = 1,row = 3) 
window.mainloop() 