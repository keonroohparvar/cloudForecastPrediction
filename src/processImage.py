import os
import shutil
from PIL import Image
import cv2
import numpy as np
from openpyxl import Workbook, workbook
from openpyxl.utils import get_column_letter

workbook = Workbook()
sheet = workbook.active

# This file will process JPEG images using the opencv package


# Helper Function that returns a list of imagePath strings that are the images in the ./images folder
def loadImages(folderName):
    images = []

    numImages = len([name for name in os.listdir(f"./{folderName}")])

    # print(f"numImages is: {numImages}")

    for i in range(numImages):
        baseImg = f"img{i}.jpg"
        imgPath = os.path.join(folderName, baseImg)
        images.append(imgPath)

    return images


# Find Sun by comparing image to six differnet template images of the sun
def findSun(imagePath, templateImagesPath):
    # print("Finding Sun for Image {}...".format(imagePath))

    # Checks that dir is there
    if not os.path.isdir(templateImagesPath):
        print("Template images dir not correctly formatted.")

    # Open image
    image = cv2.imread(imagePath, 0)
    # print("Original Image Shape: {}".format(image.shape))

    # Array to store image names
    imgNames = []
    for filename in os.listdir(templateImagesPath):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            imgNames.append(os.path.join(templateImagesPath, filename))

    # We will be using opencv's template matching to find the Sun. we will first compare the image to
    # template1.jpg, which is a clear view of the sun. We'll then go through and compare each template until we
    # find one that is pretty close.

    # Variable for formatting which template we are looking at
    # imgNum = i+1
    
    # Format image name to: {cwd}/src/template{imgNum}.jpg
    templateName = imgNames[0]
    templateName = templateName[:len(templateName) - 5]
    templateName = templateName + "1.jpg"

    # Read template Image
    template = cv2.imread(templateName, 0)

    # Create copy of original image to edit
    imgBox = image.copy()

    # Apply template matching to image with specified template
    result = cv2.matchTemplate(imgBox, template, cv2.TM_CCORR)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Assign the maximum equal value of template to variable topleft location
    topLeft = max_loc

    # Find dimensions of template image for drawing the box
    width, height = template.shape[::-1]

    # Calculate Dimensions of Box around object
    bottomRight = (topLeft[0] + width, topLeft[1] + height)

    # Find middle point of Sun
    middle = (int(topLeft[0]+(width/2)), int(topLeft[1] + (height / 2)))

    # Draw Box around object
    cv2.rectangle(imgBox, topLeft, bottomRight, 0, 2)

    # Draw Circle around Middle
    cv2.circle(imgBox, middle, 5, (0, 0, 0), -1)

    # Resize imgBox for better viewing
    newWidth = int(imgBox.shape[1]*50 / 100)
    newHeight = int(imgBox.shape[0]*50/100)
    smallerImg = cv2.resize(imgBox, (newWidth, newHeight))


    # print("Value is: {} and width:{} and height:{}".format(max_val, width, height))

    # print("Max location: ({},{})".format(max_loc[0], max_loc[1]))
    # print("Min Location: ({}, {})".format(min_loc[0], min_loc[1]))

    #cv2.imshow("Test", smallerImg)

    #cv2.waitKey()

    return middle


# Function to create all 7 rings which we will be checking the haze index %'s of
def createRings(imagePath, middle):
    # Opening Image
    image = cv2.imread(imagePath, 0)
    imageColor = cv2.imread(imagePath, 1)

    # Parse height and width from image, and create empty masks for 7 rings
    height, width = image.shape
    # print("img: {} x {}".format(width, height))
    mask1 = np.zeros((height, width), np.uint8)
    mask2 = np.zeros((height, width), np.uint8)
    mask3 = np.zeros((height, width), np.uint8)
    mask4 = np.zeros((height, width), np.uint8)
    mask5 = np.zeros((height, width), np.uint8)
    mask6 = np.zeros((height, width), np.uint8)
    mask7 = np.zeros((height, width), np.uint8)

    # Value to determine Ring Seperation
    ringPixels = 200
    centerPixels = 70

    # Drawing Rings
    circle1Img = cv2.circle(mask1, middle, centerPixels, (255, 255, 255), thickness=-1)
    circle2Img = cv2.circle(mask2, middle, centerPixels + (1 * ringPixels), (255, 255, 255), thickness=-1)
    circle3Img = cv2.circle(mask3, middle, centerPixels + (2 * ringPixels), (255, 255, 255), thickness=-1)
    circle4Img = cv2.circle(mask4, middle, centerPixels + (3 * ringPixels), (255, 255, 255), thickness=-1)
    circle5Img = cv2.circle(mask5, middle, centerPixels + (4 * ringPixels), (255, 255, 255), thickness=-1)
    circle6Img = cv2.circle(mask6, middle, centerPixels + (5 * ringPixels), (255, 255, 255), thickness=-1)
    circle7Img = cv2.circle(mask7, middle, centerPixels + (6 * ringPixels), (255, 255, 255), thickness=-1)

    # Making inverses of each mask to 'Bitwise AND' with future Rings to seperate each ring. For example, Ring 5 should NOT contain any pixels from Ring 4, so
    # we will AND the Ring 5 mask with the inverse of the Ring 4 mask.
    mask1Inv = cv2.bitwise_not(mask1)
    ring1Mask = mask1

    mask2Inv = cv2.bitwise_not(mask2)
    ring2Mask = cv2.bitwise_and(circle2Img, mask1Inv)

    mask3Inv = cv2.bitwise_not(mask3)
    ring3Mask = cv2.bitwise_and(circle3Img, mask2Inv)

    mask4Inv = cv2.bitwise_not(mask4)
    ring4Mask = cv2.bitwise_and(circle4Img, mask3Inv)

    mask5Inv = cv2.bitwise_not(mask5)
    ring5Mask = cv2.bitwise_and(circle5Img, mask4Inv)

    mask6Inv = cv2.bitwise_not(mask6)
    ring6Mask = cv2.bitwise_and(circle6Img, mask5Inv)

    ring7Mask = cv2.bitwise_and(circle7Img, mask6Inv)
    
    # Create masked rings for all 7 rings by 'Bitwise AND'-ing each mask with the inverse of the previous mask.
    maskedData1 = cv2.bitwise_and(imageColor, imageColor, mask=circle1Img)
    maskedData2 = cv2.bitwise_and(imageColor, imageColor, mask=ring2Mask)
    maskedData3 = cv2.bitwise_and(imageColor, imageColor, mask=ring3Mask)
    maskedData4 = cv2.bitwise_and(imageColor, imageColor, mask=ring4Mask)
    maskedData5 = cv2.bitwise_and(imageColor, imageColor, mask=ring5Mask)
    maskedData6 = cv2.bitwise_and(imageColor, imageColor, mask=ring6Mask)
    maskedData7 = cv2.bitwise_and(imageColor, imageColor, mask=ring7Mask) 

    # Determine how many pixels are in each ring
    ringPixels = []
    ringPixels.append(np.count_nonzero(ring1Mask))
    ringPixels.append(np.count_nonzero(ring2Mask))
    ringPixels.append(np.count_nonzero(ring3Mask))
    ringPixels.append(np.count_nonzero(ring4Mask))
    ringPixels.append(np.count_nonzero(ring5Mask))
    ringPixels.append(np.count_nonzero(ring6Mask))
    ringPixels.append(np.count_nonzero(ring7Mask))

    # Create array of Ring Masks to use in the processImage() function
    ringMasks = []
    ringMasks.append(ring1Mask)
    ringMasks.append(ring2Mask)
    ringMasks.append(ring3Mask)
    ringMasks.append(ring4Mask)
    ringMasks.append(ring5Mask)
    ringMasks.append(ring6Mask)
    ringMasks.append(ring7Mask)
    ringMasks.append(ring7Mask)

    # print(ringPixels)


    # cv2.imshow("Ring 5 mask", ring5Mask)

    cv2.waitKey()
    
    return [maskedData1, maskedData2, maskedData3, maskedData4, maskedData5, maskedData6, maskedData7], ringPixels, ringMasks


# Function to process images and calculate % values for each ring.
def processImage(imagePath, thresholdLow, thresholdHigh, printImg=False):
    # Open Original Image
    # originalImage = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
    # print(f"imagePath is {imagePath}")
    originalImage = cv2.imread(imagePath, 1)

    # Finding Sun
    middle = findSun(imagePath, './src/templateImages')

    # Creating Rings and Counting the Pixels in each Ring
    rings, ringPixels, ringMasks = createRings(imagePath, middle)

    # Create Variable for Storing Percentages
    percentages = [0, 0, 0, 0, 0, 0, 0]

    finalImg = np.zeros(originalImage.shape, np.uint8)

    # Handles ring % calculation for each indiviual ring and saves it into percentages array
    for i in range(len(rings)):
        ring = rings[i]

        # cv2.imshow('ring{}'.format(i), ring)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        
        # Parse Ring into appropriate channels
        blueChannel = ring[:,:,0]
        greenChannel = ring[:,:,1]
        redChannel = ring[:,:,2]

        # Calculate numpy 2-d array of each pixel's haze index
        hazeChannel = (((redChannel + blueChannel) / 2) - greenChannel + 1) / (((redChannel + blueChannel) / 2) + greenChannel + 1)

        # Create mask using threshold values
        currentRingMask = cv2.inRange(hazeChannel, thresholdLow, thresholdHigh)

        # Need to 'Bitwise AND' each ring mask with the inverse of the mask above and below it to only leave the pixels in the specified ring
        currentRingMask = cv2.bitwise_and(currentRingMask, currentRingMask, mask=ringMasks[i])

        # See How many Pixels are Passed the Threshold in ringMask
        currentRingPixels = np.count_nonzero(currentRingMask)
        # print("Ring {} has {} nonzero pix out of {}, yielding {}%".format(i, currentRingPixels, ringPixels[i-1], (currentRingPixels / ringPixels[i-1])))

        # Calculate % covered and store in percentages
        percentages[i] = 100 - 100 * (currentRingPixels / ringPixels[i-1])
        percentages[i] = 0 if percentages[i] < 0 else percentages[i]


        frame = cv2.bitwise_and(originalImage, originalImage, mask=currentRingMask)

        # print("Final image shape: {}".format(finalImg.shape))
        # print("Frame Shape: {}".format(frame.shape))

        finalImg = cv2.add(finalImg, frame, 1)

    return percentages

        
    # Display the final Image
    if printImg:
        newWidth = int(finalImg.shape[1] * 50 / 100)
        newHeight = int(finalImg.shape[0] * 50 / 100)
        smallerFinalImg = cv2.resize(finalImg, (newWidth, newHeight))
        cv2.imshow("Final Image with threshold {} to {}".format(thresholdLow, thresholdHigh), smallerFinalImg)
        print("Percentages for Image {} are: {}".format(imagePath, percentages))
        cv2.waitKey()
        cv2.destroyAllWindows()


# Function to change names of Columns in worksheet 
def updateWorksheetNames():
    global sheet

    sheet["A1"] = "Ring 1 (Sun)"
    sheet["B1"] = "Ring 2"
    sheet["C1"] = "Ring 3"
    sheet["D1"] = "Ring 4"
    sheet["E1"] = "Ring 5"
    sheet["F1"] = "Ring 6"
    sheet["G1"] = "Ring 7"



# Function to save percentages in appropriate excell column
def savePercentages(percentages, rowNum):
    global sheet

    # This addition is to account for the 1st row being the Column Names 
    rowNum += 2

    
    sheet[f"A{rowNum}"] = percentages[0]
    sheet[f"B{rowNum}"] = percentages[1]
    sheet[f"C{rowNum}"] = percentages[2]
    sheet[f"D{rowNum}"] = percentages[3]
    sheet[f"E{rowNum}"] = percentages[4]
    sheet[f"F{rowNum}"] = percentages[5]
    sheet[f"G{rowNum}"] = percentages[6]


# Main Method
if __name__ == "__main__":
    
    print("Welcome to the process Image shell.")
    currOption = ""
    
    while currOption != "q":
        # Print options for user in Shell
        print("\nOptions:\n\tp folderPath --- Process all images in specified folderPath")
        print("\td --- Display potential folder paths for processing")
        print("\tq --- Quit the shell\n")
        
        # Parse input
        rawInput = input("\033[1;32;40mprocessshell\033[1;0;0m$ ")
        inputList = rawInput.split(" ")
        currOption = inputList[0]
        if len(inputList) > 0:
            params = inputList[1:]  
        else:
            params = ["1"]
        
        # Handle p option used for processing the data at CWD + /folderPath
        if currOption == "p":
            folderPath = params[0]
            
            if folderPath == "1":
                print("Using folerpath images/2021/3-4...")
                folderPath = 'images/2021/3-4'

            if not os.path.isdir(folderPath):
                print("Error - specified folderpath is not a directory. Use option d to see available folderpaths.")
            else:
                images = loadImages(folderPath)
                print('Length of images to be processed: %d' % len(images))
                
                # Update column names to Ring 0, Ring 1, etc.
                updateWorksheetNames()

                # Iterate over each image and save its percentages to excel sheet
                for i in range(len(images)):
                    print("Processing image {}...".format(i))
                    percentages = processImage(images[i], 0.012, 0.1, False)
                    savePercentages(percentages, i)
                
                # Save excell sheet to ../trainingData/ folder with its respective file name
                folderPath = folderPath.split("/")
                folderPath.pop(0)
                newFileName = "-".join(folderPath)
                workbook.save(filename=f"./trainingData/{newFileName}.xlsx")


        # Handles the t option used for testing
        elif currOption == "t":
            testImagePath = './images/2021/testImgs/img19.jpg'

            # Console message for User
            print("---------------\nEnter 'q' to quit into either threshold value.")
            threshLow = input('Enter the threshold Low Limit: ')
            threshHigh = input('Enter the threshold High Limit: ')
            interval = input('Enter the interval that either the high or low will increment by: ')
            numIterations = input('Enter the number of iterations: ')
            lowOrHigh = input("Type 'l' for incrementing on the low threshold. Type 'h' for incrementing on the high threshold: ")
            while threshLow != "q" and threshHigh != "q":
                try:
                    threshLow = float(threshLow)
                    threshHigh = float(threshHigh)
                    numIterations = int(numIterations)
                    print("Num Iter: {}".format(numIterations))
                    interval = float(interval)

                    for i in range(numIterations):
                        tempThreshLow = threshLow
                        tempThreshHigh = threshHigh
                        if lowOrHigh == "l":
                            tempThreshLow = threshLow + (interval * i)
                        elif lowOrHigh == "h":
                            tempThreshHigh = threshHigh + (interval * i)

                        processImage(testImagePath, tempThreshLow, tempThreshHigh, True)
                    
                    cv2.waitKey()
                    cv2.destroyAllWindows()

                except Exception as e:
                    print("\nERROR:\n")
                    print(e)
                    print("\nCould not properly parse input. Quitting...\n")
                    threshLow = "q"
                
                print("---------------\nEnter 'q' to quit into either threshold value.")
                threshLow = input('Enter the threshold Low Limit: ')
                threshHigh = input('Enter the threshold High Limit: ')
                interval = input('Enter the interval that either the high or low will increment by: ')
                numIterations = input('Enter the number of iterations: ')
                lowOrHigh = input("Type 'l' for incrementing on the low threshold. Type 'h' for incrementing on the high threshold: ")


        # Handles the d option used for displaying folders
        elif currOption == "d":
            pass


        # Handles unknown flags
        else:
            print("Unknown option. Try again.\n")
    
    
    exit(0)    

