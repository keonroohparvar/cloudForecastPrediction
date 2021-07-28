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


# Function to create all 4 rings which we will be checking the haze index %'s of
def createRings(imagePath, middle, activeMask):
    # Opening Image
    image = cv2.imread(imagePath, 0)
    imageColor = cv2.imread(imagePath, 1)

    # Parse height and width from image, and create empty masks for 7 rings
    height, width = image.shape
    mask1 = np.zeros((height, width), np.uint8)
    mask2 = np.zeros((height, width), np.uint8)
    mask3 = np.zeros((height, width), np.uint8)
    mask4 = np.zeros((height, width), np.uint8)

    # Create masks for each direction of ring (NW, NE, SE, SW)
    maskNW = np.zeros((height, width), np.uint8)
    maskNE = np.zeros((height, width), np.uint8)
    maskSE = np.zeros((height, width), np.uint8)
    maskSW = np.zeros((height, width), np.uint8)

    cv2.rectangle(maskNW, (0, 0), middle, 255, thickness=-1)
    cv2.rectangle(maskNE, middle, (width, 0), 255, thickness=-1)
    cv2.rectangle(maskSE, middle, (width, height), 255, thickness=-1)
    cv2.rectangle(maskSW, (0, height), middle, 255, thickness=-1)

    dirRings = []

    dirRings.append(maskNW)
    dirRings.append(maskNE)
    dirRings.append(maskSE)
    dirRings.append(maskSW)

    # Value to determine Ring Seperation
    ringPixels = 350
    centerPixels = 150

    # Drawing Rings
    circle1Img = cv2.circle(mask1, middle, centerPixels, (255, 255, 255), thickness=-1)
    circle2Img = cv2.circle(mask2, middle, centerPixels + (1 * ringPixels), (255, 255, 255), thickness=-1)
    circle3Img = cv2.circle(mask3, middle, centerPixels + (2 * ringPixels), (255, 255, 255), thickness=-1)
    circle4Img = cv2.circle(mask4, middle, centerPixels + (3 * ringPixels), (255, 255, 255), thickness=-1)

    # Making inverases of each mask to 'Bitwise AND' with future Rings to seperate each ring. For example, Ring 5 should NOT contain any pixels from Ring 4, so
    # we will AND the Ring 5 mask with the inverse of the Ring 4 mask.
    mask1Inv = cv2.bitwise_not(mask1)
    ring1Mask = mask1

    mask2Inv = cv2.bitwise_not(mask2)
    ring2Mask = cv2.bitwise_and(circle2Img, mask1Inv)

    mask3Inv = cv2.bitwise_not(mask3)
    ring3Mask = cv2.bitwise_and(circle3Img, mask2Inv)

    mask4Inv = cv2.bitwise_not(mask4)
    ring4Mask = cv2.bitwise_and(circle4Img, mask3Inv)


    # Mask each ring with itself, but eliminate the black ring around the image
    ring1Mask = cv2.bitwise_and(activeMask, activeMask, mask=ring1Mask)
    ring2Mask = cv2.bitwise_and(activeMask, activeMask, mask=ring2Mask)
    ring3Mask = cv2.bitwise_and(activeMask, activeMask, mask=ring3Mask)
    ring4Mask = cv2.bitwise_and(activeMask, activeMask, mask=ring4Mask)


    # Create masked rings for all 7 rings by 'Bitwise AND'-ing each mask with the inverse of the previous mask.
    maskedData1 = cv2.bitwise_and(imageColor, ring1Mask)
    maskedData2 = cv2.bitwise_and(imageColor, ring2Mask)
    maskedData3 = cv2.bitwise_and(imageColor, ring3Mask)
    maskedData4 = cv2.bitwise_and(imageColor, ring4Mask)

    # cv2.destroyAllWindows()
    # newWidth = int(maskedData4.shape[1]*25 / 100)
    # newHeight = int(maskedData4.shape[0]*25/100)
    # smallerImg = cv2.resize(maskedData5, (newWidth, newHeight))
    # cv2.imshow("Circle 5 Img", smallerImg)
    # cv2.waitKey()
    # exit()

    # Determine how many pixels are in each ring
    ringPixels = []
    ringPixels.append(np.count_nonzero(ring1Mask))
    ringPixels.append(np.count_nonzero(ring2Mask))
    ringPixels.append(np.count_nonzero(ring3Mask))
    ringPixels.append(np.count_nonzero(ring4Mask))
    # Create array of Ring Masks to use in the processImage() function
    ringMasks = []
    ringMasks.append(ring1Mask)
    ringMasks.append(ring2Mask)
    ringMasks.append(ring3Mask)
    ringMasks.append(ring4Mask)

    # print(ringPixels)


    # cv2.imshow("Ring 5 mask", ring5Mask)

    # cv2.waitKey()
    
    return [maskedData1, maskedData2, maskedData3, maskedData4], ringPixels, ringMasks, dirRings


# Function to process images and calculate % values for each ring.
def processImage(imagePath, thresholdLow, thresholdHigh, printImg=False):
    # Open Original Image
    # originalImage = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
    print(f"imagePath is {imagePath}")
    originalImage = cv2.imread(imagePath, 1)

    # Finding Sun
    middle = findSun(imagePath, './src/templateImages')

    # Create active mask
    activeMask = np.zeros(originalImage.shape, np.uint8)
    center = (int(originalImage.shape[1] / 2), int(originalImage.shape[0]/2))
    cv2.circle(activeMask, center, 1000, (255, 255, 255), thickness=-1 )
    

    # Creating Rings and Counting the Pixels in each Ring
    rings, ringPixels, ringMasks, dirRings = createRings(imagePath, middle, activeMask)

    # Create Variable for Storing Percentages
    percentages = [[], [], [], []]

    finalImg = np.zeros(originalImage.shape, np.uint8)

    # Handles ring % calculation for each indiviual ring and saves it into percentages array
    for i in range(len(rings)):
        
        ring = rings[i]
        currRingMask = ringMasks[i]

        # # Print Current Ring Ring
        # newWidth = int(ring.shape[1]*25 / 100)
        # newHeight = int(ring.shape[0]*25/100)
        # smallerImg = cv2.resize(ring, (newWidth, newHeight))
        # cv2.imshow("Curr Ring", smallerImg)

        
        # Parse Ring into appropriate channels
        blueChannel = ring[:,:,0]
        greenChannel = ring[:,:,1]
        redChannel = ring[:,:,2]

        # Calculate numpy 2-d array of each pixel's haze index
        hazeChannel = (((redChannel + blueChannel) / 2) - greenChannel + 1) / (((redChannel + blueChannel) / 2) + greenChannel + 1)

        # Create mask using threshold values
        currentCloudMask = cv2.inRange(hazeChannel, thresholdLow, thresholdHigh)

        # # Print Current Ring Mask
        # newWidth = int(currentCloudMask.shape[1]*25 / 100)
        # newHeight = int(currentCloudMask.shape[0]*25/100)
        # smallerImg = cv2.resize(currentCloudMask, (newWidth, newHeight))
        # cv2.imshow("Curr Ring Mask", smallerImg)
            

        # Need to 'Bitwise AND' each ring mask with the inverse of the mask above and below it to only leave the pixels in the specified ring
        currentCloudMask = cv2.bitwise_and(currRingMask, currRingMask, mask=currentCloudMask)

        # Partition ring data into 4 parts (NW, NE, SE, SW)
        currNWRing = cv2.bitwise_and(currentCloudMask, currentCloudMask, mask=dirRings[0])
        currNERing = cv2.bitwise_and(currentCloudMask, currentCloudMask, mask=dirRings[1])
        currSERing = cv2.bitwise_and(currentCloudMask, currentCloudMask, mask=dirRings[2])
        currSWRing = cv2.bitwise_and(currentCloudMask, currentCloudMask, mask=dirRings[3])


        # # Print Current NE Mask
        # newWidth = int(currNWRing.shape[1]*25 / 100)
        # newHeight = int(currNWRing.shape[0]*25/100)
        # smallerImg = cv2.resize(currNWRing, (newWidth, newHeight))
        # cv2.imshow("NW Ring", smallerImg)


        # See How many Pixels are Inside the Threshold in each ring (currNWRing, currNERing, etc.)
        currNWPixels = np.count_nonzero(currNWRing)
        currNEPixels = np.count_nonzero(currNERing)
        currSEPixels = np.count_nonzero(currSERing)
        currSWPixels = np.count_nonzero(currSWRing)
        # print("Ring {} has {} nonzero pix out of {}, yielding {}%".format(i, currentRingPixels, ringPixels[i-1], s(currentRingPixels / ringPixels[i-1])))


        # Determine how many pixels there are total in each ring mask
        NWMaskPixels = np.count_nonzero(cv2.bitwise_and(currRingMask, currRingMask, mask=dirRings[0]))
        NEMaskPixels = np.count_nonzero(cv2.bitwise_and(currRingMask, currRingMask, mask=dirRings[1]))
        SEMaskPixels = np.count_nonzero(cv2.bitwise_and(currRingMask, currRingMask, mask=dirRings[2]))
        SWMaskPixels = np.count_nonzero(cv2.bitwise_and(currRingMask, currRingMask, mask=dirRings[3]))
        
        # Calculate % covered and store in percentages for each ring (NW, NE, SE, SW)
        currNWPercent = 0 if NWMaskPixels == 0 else 100 - 100 * (currNWPixels / NWMaskPixels)
        currNEPercent = 0 if NEMaskPixels == 0 else 100 - 100 * (currNEPixels / NEMaskPixels)
        currSEPercent = 0 if SEMaskPixels == 0 else 100 - 100 * (currSEPixels / SEMaskPixels)
        currSWPercent = 0 if SWMaskPixels == 0 else 100 - 100 * (currSWPixels / SWMaskPixels)
        thesePercentages = [currNWPercent, currNEPercent, currSEPercent, currSWPercent]
        for j in range(len(thesePercentages)):
            if thesePercentages[j] < 0:
                thesePercentages[j] = 0

        # print(f"NW Percentage is {currNWPercent}")

        percentages[i] = thesePercentages

        frame = cv2.bitwise_and(originalImage, currentCloudMask)

        # print("Final image shape: {}".format(finalImg.shape))
        # print("Frame Shape: {}".format(frame.shape))

        finalImg = cv2.add(finalImg, frame, 1)


        

    # Display the final Image if printImg is True
    if printImg:
        newWidth = int(finalImg.shape[1] * 50 / 100)
        newHeight = int(finalImg.shape[0] * 50 / 100)
        smallerFinalImg = cv2.resize(finalImg, (newWidth, newHeight))
        cv2.imshow("Final Image with threshold {} to {}".format(thresholdLow, thresholdHigh), smallerFinalImg)
        cv2.waitKey()
        print("Percentages for Image {} are: {}".format(imagePath, percentages))
        # cv2.waitKey()
        # cv2.destroyAllWindows()

    return percentages


# Function to change names of Columns in worksheet 
def updateWorksheetNames():
    global sheet

    sheet["A1"] = "Ring 1 (Sun) NW"
    sheet["B1"] = "Ring 1 (Sun) NW"
    sheet["C1"] = "Ring 1 (Sun) NW"
    sheet["D1"] = "Ring 1 (Sun) NW"
    sheet["E1"] = "Ring 2 NW"
    sheet["F1"] = "Ring 2 NE"
    sheet["G1"] = "Ring 2 SE"
    sheet["H1"] = "Ring 2 SW"
    sheet["I1"] = "Ring 3 NW"
    sheet["J1"] = "Ring 3 NE"
    sheet["K1"] = "Ring 3 SE"
    sheet["L1"] = "Ring 3 SW"
    sheet["M1"] = "Ring 4 NW"
    sheet["N1"] = "Ring 4 NE"
    sheet["O1"] = "Ring 4 SE"
    sheet["P1"] = "Ring 4 SW"



# Function to save percentages in appropriate excell column
def savePercentages(percentages, rowNum):
    global sheet

    # This addition is to account for the 1st row being the Column Names 
    rowNum += 2

    
    sheet[f"A{rowNum}"] = percentages[0][0]
    sheet[f"B{rowNum}"] = percentages[0][1]
    sheet[f"C{rowNum}"] = percentages[0][2]
    sheet[f"D{rowNum}"] = percentages[0][3]
    sheet[f"E{rowNum}"] = percentages[1][0]
    sheet[f"F{rowNum}"] = percentages[1][1]
    sheet[f"G{rowNum}"] = percentages[1][2]
    sheet[f"H{rowNum}"] = percentages[1][3]
    sheet[f"I{rowNum}"] = percentages[2][0]
    sheet[f"J{rowNum}"] = percentages[2][1]
    sheet[f"K{rowNum}"] = percentages[2][2]
    sheet[f"L{rowNum}"] = percentages[2][3]
    sheet[f"M{rowNum}"] = percentages[3][0]
    sheet[f"N{rowNum}"] = percentages[3][1]
    sheet[f"O{rowNum}"] = percentages[3][2]
    sheet[f"P{rowNum}"] = percentages[3][3]


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

