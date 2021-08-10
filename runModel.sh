#!/bin/bash

rm epochData.txt

lineCount=60
lossSum=0
valLossSum=0
for (( i=1; i<=$1; i++))
   do 
      python src/weatherModel.py >> epochData.txt
      
      loss=$(awk -v var="$lineCount" 'NR==var {print $18}' epochData.txt)
      lossSum=$(echo "$lossSum" + "$loss" | bc -l)

      valLoss=$(awk -v var="$lineCount" 'NR==var {print $24}' epochData.txt)
      valLossSum=$(echo "$valLossSum" + "$valLoss" | bc -l)
      lineCount=$(expr $lineCount + 60)

      if [ "$#" == "2" ] && [ "$2" == "p" ]
      then
         echo " $i Trial Loss: $loss" 
         echo " $i Trial Loss: $valLoss"
      fi 

      echo "-----------------------------------------------"
   done

printf '\nAverage Loss: '
awk "BEGIN {print $lossSum/$i}"

printf "\nAverage Validation Loss: "
awk "BEGIN {print $valLossSum/$i}"