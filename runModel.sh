#!/bin/bash

rm epochData.txt
rm SummaryResults.txt

lineCount=60
lossSum=0
valLossSum=0
for i in 1 2 3 4 5 6
   do 
      python src/weatherModel.py >> epochData.txt

      loss=$(awk -v var="$lineCount" 'NR==var {print $18}' epochData.txt)
      lossSum=$(echo "$lossSum" + "$loss" | bc -l)

      valLoss=$(awk -v var="$lineCount" 'NR==var {print $24}' epochData.txt)
      valLossSum=$(echo "$valLossSum" + "$valLoss" | bc -l)
      lineCount=$(expr $lineCount + 60)
      
      echo "-------------------------------"
   done

printf '\nAverage Loss: '
awk "BEGIN {print $lossSum/6}"

printf "\nAverage Validation Loss: "
awk "BEGIN {print $valLossSum/6}"