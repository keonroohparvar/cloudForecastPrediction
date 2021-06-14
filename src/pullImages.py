# This script pulls jpeg images from the below webpage:
# https://www.cameraftp.com/Camera/Cameraplayer.aspx?parentID=229229999&shareID=14125452
# and stores them in the ./images folder.

from urllib import request
import os
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup



# Camera Images URL
mainURL = "https://www.cameraftp.com/Camera/Cameraplayer.aspx?parentID=229229999&shareID=14125452"
dayURL = "https://www.cameraftp.com/camera/CameraPlayerMultiHours.htm?cameraID=229229999&name=Sky_Camera&shareID=14125452&start=2021-04-19%2021:09:33"


# Function to handle the input to the shell
def parseInput(inputStr):
    inputList = inputStr.split(" ")
    
    # Handles -p date formatting
    if inputList[0] == "p":
        correctDate = True
        if len(inputList) != 4:
            correctDate = False
        try:
            # Convert string numbers to integer types
            for i in range(1,4):
                inputList[i] = int(inputList[i])

            # Confirm date is a valid date
            if inputList[1] > 12:
                correctDate = False
            elif inputList[2] > 31:
                correctDate = False
            elif len(str(inputList[3])) == 2:
                tempYear = str(inputList[3])
                tempYear = "20" + tempYear
                inputList[3] = int(tempYear)
            
            return (correctDate, inputList)

        except:
            print("Error in date formatting.")
            correctDate = False
            return (correctDate, inputList)
    
    # Handles -q option
    elif inputList[0] == "q":
        return (True, inputList)

    else:
        return (False, inputList)
    


# Format driver to specific time
def formatDriver(day, month, year):
    try :
        global driver
        driver.get(dayURL)

        # Wait for page to update
        element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "datepickerFrom"))
        )
        
        # Format dateFrom HTML box
        dateFromClass = driver.find_element_by_xpath("/html/body/div[2]/ul/li[3]")
        dateFrom = driver.find_element_by_id("datepickerFrom")
        dateFromText = "{} {} {} 10:00 AM".format(str(day), str(month), str(year))

        # Put in raw text data into dateFrom HTML forms
        dateFrom.clear()
        dateFrom.send_keys(dateFromText)
        dateFrom.send_keys(Keys.TAB)
        time.sleep(1)

        # Format dateTo HTML box
        dateTo = driver.find_element_by_id("datepickerTo")
        dateToText = "{} {} {} 6:00PM".format(str(day), str(month), str(year))

        # Send raw text data into dateTo text box
        dateTo.clear()
        time.sleep(.5)
        dateTo.click()
        time.sleep(1)
        dateTo.send_keys(Keys.CONTROL, 'a')
        dateTo.send_keys(dateToText)
        time.sleep(1)
        dateTo.send_keys(Keys.TAB)

        # Change layout of images to 1x1 so that there is only one image frame that goes from 8:00 AM - 8:00 PM
        layout1 = driver.find_element_by_xpath("/html/body/div[2]/ul/li[5]/input[1]")
        layout2 = driver.find_element_by_xpath("/html/body/div[2]/ul/li[5]/input[2]")
        layout1.send_keys(Keys.CONTROL, 'a')
        layout1.send_keys("1")
        layout2.send_keys(Keys.CONTROL, 'a')
        layout2.send_keys("1")

        # Update image resolution to 640*480
        resolutionButton = driver.find_element_by_id("Button1")
        resolutionButton.click()
        option640 = driver.find_element_by_xpath("/html/body/div[2]/ul/li[6]/div/ul/li[2]/a")
        option640.click()

        # Find applyButton and click it to update elements
        applyButton = driver.find_element_by_xpath("/html/body/div[2]/ul/li[7]/a")
        applyButton.click()

        time.sleep(1)
        
        

        return True

    except Exception as e:
        print("Error in formatting driver - Error:\n{}".format(e))
        driver.quit()
        return False


# Pulls images from the afternoon of the day in the format: day/month/year
def pullDayImages(day, month, year, maxImgs, driverFormatted):
    # Declaring driver as global to edit its value
    global driver

    if driverFormatted == False:
        print("The driver has not been formatted with a date.")
        driver.quit()
        return -1

    print("\nGoing to pull images...")
    
    
    # Creating folder for storage of the form CWD/year/daymonth
    #   Example: the date 03/04/2021 will produce cwd/2021/0304
    if not os.path.isdir('images/{}/{}'.format(str(year), str(day)+"-"+str(month))):
        os.makedirs('images/{}/{}'.format(str(year), str(day)+"-"+str(month)))
    
    # Clicking the Play button twice to pause, and then play
    playButton = driver.find_element_by_xpath("/html/body/div[2]/ul/li[2]/span")
    playButton.click()
    time.sleep(.25)
    playButton.click()
    
    # Getting Image Sources
    try:
        driver.switch_to.frame(1)
        element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "cameraImg"))
        )
    except:
        print("Error with webdriver interfacing the page.") 
        driver.quit()
        return -1

    # Creating variables to store the number of images, camera WebElement, and things for iterating over the loop
    numImgsSaved = 0
    cameraImg = driver.find_element_by_id("cameraImg")
    prevDataURL = cameraImg.get_attribute("src")
    currentDataURL = ""

    # Loop to save images. Exits when either there have been maxImgs number of images saved, or the image source is
    # no longer changing.
    while (int(numImgsSaved) != int(maxImgs)) and (prevDataURL != currentDataURL):
        print("Loop num {}".format(numImgsSaved))
        currentDataURL = cameraImg.get_attribute("src")

        # Pull current image data and write it to file under its respective folder
        with request.urlopen(currentDataURL) as response:
            data = response.read()
        with open(os.getcwd() + "/images/{}/{}/img{}.jpg".format(str(year), str(day)+"-"+str(month), str(numImgsSaved)), "wb") as f:
            f.write(data)

        # Loop to wait for current image data to change, or simply wait 5 seconds which indicates there are no images
        halfSecondCount = 0
        prevDataURL = currentDataURL
        while currentDataURL == prevDataURL and halfSecondCount <= 10:
            cameraImg = driver.find_element_by_id("cameraImg")
            currentDataURL = cameraImg.get_attribute("src")
            halfSecondCount += 1
            time.sleep(.5)
        
        # Increment count of images saved
        numImgsSaved += 1
        

    # Revert back to parent frame
    time.sleep(1)
    driver.switch_to.default_content()
    print("Done saving images!")

    driver.quit()


# Main function
if __name__ == "__main__":
    welcomeMessage = "\nWelcome to the automated script for pulling images from the camera.\nThe URL for the camera is:\n\n"
    welcomeMessage += mainURL
    print(welcomeMessage)
    currOption = ""
    while currOption != "q":
        # Print option message and shell output
        print("\nOptions:\n\tp day month year --- Pull images from Camera from 9AM onwards at the date day/month/year")
        print("\tq --- Quit the shell\n")
        currOption = input("\033[1;32;40mpullshell\033[1;0;0m$ ")
        
        # Parse input into the form (validInp, inputList) where: 
        #   validInp: a boolean representing the input being valid
        #   inputList: list representing the input. inputList[0] is the option, and the remainder are the parameters
        inputTuple = parseInput(currOption)
        
        # Checks if the input is valid
        if (inputTuple[0]) == True:
            option = inputTuple[1][0]
            params = inputTuple[1][1:]

            # Pull pictures
            if option == "p":
                
                # Formatting the max number of images from input
                numImages = "k"
                while not numImages.isdigit() and numImages != "-1":
                    numImages = input("What is the max number of images you would like to save? (-1 means no limit): ")
                print("Pulling photos from {}/{}/{} 8:00 AM...\n".format(params[0], params[1], params[2]))

                global driver
                driver = webdriver.Chrome()
                
                # Formatting the driver with the desired date
                driverFormatted = formatDriver(params[0], params[1], params[2])

                # Calling the function to pull images
                pullDayImages(params[0], params[1], params[2], numImages, driverFormatted)

            # Quit Shell
            elif option == "q":
                print("Quitting...")

            # Invalid Input
            else:
                print("Error - Unknown input. Try again.")
        
        # Statement for if the input string does not pass parseInput()
        else:
            print("Could not properly format input. Try again.\n")



    
