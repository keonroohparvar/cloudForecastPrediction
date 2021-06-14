# This file is for storing old functions that have a newer version.


# Calculate each pixel's sky and haze index; check to see if they surpass
# the threshold value. If they do, update them to Blue. If not, update to
# Black.
"""
def processImage(imagePIL, imageCV, threshold, folderPath):
    # Make copy of image that is grey
    grayImg = cv2.cvtColor(imageCV, cv2.COLOR_BGR2GRAY)

    time.sleep(2)
    print(grayImg)

    # cv2.imshow("Window", grayImg)

    # Find the brightest pixel (The sun)
    (maxPixelX, maxPixelY) = findSun(grayImg)

    # Draw ring around Sun 
    cv2.circle(imageCV, (maxPixelX, maxPixelY), 30, (255, 0, 0), 10)

    img2 = cv2.resize(imageCV, (400, 400))

    cv2.imshow("CircleImage", img2)

    cv2.waitKey()

    return -1
    
    # # Iterating over each pixel
    # for y in range(imagePIL.height):
    #     for x in range(imagePIL.width):
    #         hazeIndex = calculateHazeIndex(imagePIL, x, y)
    #         if (hazeIndex < threshold):
    #             imagePIL.putpixel((x,y), (255, 0, 0))
    #         elif (hazeIndex < (threshold + 0.005)):
    #             imagePIL.putpixel((x,y), (0, 255, 0))
    #         else:
    #             pass

    # Formatting new image path in OS
    originalImageDirs = folderPath.split("/")
    originalImageDirs[0] = "/processedImages"
    newImagePath = "/".join(originalImageDirs)

    # Retrieving the original image name
    imageDirs = imagePIL.filename.split("/")
    imageName = imageDirs[len(imageDirs)-1]
    imageName = (imageName.split("."))[0]
    cwd = os.getcwd()

    # Creating dir if not there
    if not os.path.isdir(cwd+newImagePath):
        os.makedirs(cwd+newImagePath)

    print("Saving image:\t{}".format(imageName))
    imagePIL.save(cwd+newImagePath+"/{}.processed.jpeg".format(imageName))
    #image.show()
"""

# Find Sky Index for a single pixel
def calculateSkyIndex(image, x, y):
    currPixel = image.getpixel((x, y))
    countRed = currPixel[0]
    countBlue = currPixel[2]
    return ((countBlue - countRed) / (countBlue + countRed))

# Find Haze Index for a single pixel, which is a more accurate sky index
def calculateHazeIndex(image, x, y):    
    currPixel = image.getpixel((x, y))
    countRed = currPixel[0]
    countGreen = currPixel[1]
    countBlue = currPixel[2]
    return (((countRed + countBlue) / 2) - countGreen + 1) / (((countRed + countBlue) / 2) + countGreen + 1)
