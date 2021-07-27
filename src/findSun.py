from pysolar.solar import *
import math
import cv2
import numpy as np
import datetime
import pytz


def boundCoordinates(x, y, img):
  height, width = img.shape[:2]
  if (x >= width):
    x = width - 1
  elif (x < 0):
    x = 0
  if (y >= height):
    y = height - 1
  elif (y < 0):
    y = 0
  return (x, y)


def convertMinToTime(min):
    hour = min / 60
    minute = min % 60
    return math.floor(hour), math.floor(minute)


def getSolarAngles(imageNumber):
  startTime = 600  # 10:00am expressed in minutes
  latitude = 35.32
  longitude = -120.69
  curTime = startTime + 5 * imageNumber
  hour, minute = convertMinToTime(curTime)

  print(("0" + str(hour) if hour < 10 else str(hour)) + ":" + ("0" + str(minute) if minute < 10 else str(minute)))

  # naive date with no sense of timezone
  date = datetime.datetime(2020, 5, 19, hour, minute, 0, 0)  
  timezone = pytz.timezone("America/Los_Angeles")
  # aware date now includes timezone info (accounts for daylight savings as well)
  awareDate = timezone.localize(date)  
  azimuth = get_azimuth(latitude, longitude, awareDate)
  altitude = get_altitude(latitude, longitude, awareDate)
  return (azimuth, altitude)


def findSun(image, imageNumber):
  print("")

  azimuth, altitude = getSolarAngles(imageNumber)  # angles are in degrees
  
  print("img" + str(imageNumber))
  print("azimuth: " + str(azimuth))
  print("altitude: " + str(altitude))

  # convert to radians
  azimuth = (azimuth - 100) / 180 * math.pi
  altitude = altitude / 180 * math.pi

  height, width = image.shape[:2]
  xCenter = round(width / 2) + 10
  yCenter = round(height / 2)
  xScale = math.cos(azimuth)
  yScale = -math.sin(azimuth)
  # zScale = 1 - altitude / (math.pi / 2)  # assumes 180 degree fisheye
  zScale = 1 - altitude / (math.pi / 2 * 90 / 92.5)  # rough adjustment for 185 degree fisheye
  # distance from image edge to fisheye edge is ~250 pixels
  x = xCenter + round(xScale * zScale * (width - 250) / 2)
  y = yCenter + round(yScale * zScale * (width - 250) / 2)
  return boundCoordinates(x, y, image)


def main():
  for imageNumber in range(96):
    image = cv2.imread("./5-19/img" + str(imageNumber) + ".jpg", cv2.IMREAD_COLOR)
    sunCoordinates = findSun(image, imageNumber)
    imageCopy = image.copy()
    imageCopy = cv2.circle(imageCopy, sunCoordinates, 5, (0, 0, 0), -1)
    cv2.imshow("Test", imageCopy)
    cv2.waitKey()
    cv2.imwrite("./sunImages/img" + str(imageNumber) + "Sun" + ".jpg", imageCopy)

if __name__ == "__main__":
  main()
