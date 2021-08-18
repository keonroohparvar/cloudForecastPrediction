import requests
import pandas as pd

# query data for 5-19-20 using visual crossing API, NOTE: Need to insert own API key 
url = """https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/weatherdata/history?&combinationMethod=aggregate&aggregateMinutes=5&startDateTime=2020-05-19T10:00:00&endDateTime=2020-05-19T17:55:00&unitGroup=us&contentType=csv&location=35.32,-120.69&key={INSERT_PERSONAL_API_KEY}"""
results = requests.get(url)

# open csv file for writing
file = open("5-19-20-wind.csv", "w")

# write each line of results.text to csv file
for line in results.text:
  file.write(line)
# close csv file
file.close()

# convert csv to excel
data = pd.read_csv("5-19-20-wind.csv")
data.to_excel("5-19-20-wind.xlsx", index=None, header=True)
# generates excel file with historical weather data in 5-minute intervals on 5-19-20 at location (35.32, -120.69)
