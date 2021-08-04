#!/bin/bash

rm epochData.txt
rm SummaryResults.txt

for i in 1 2 3 4 5 6
   do 
      python src/weatherModel.py >> epochData.txt
      echo "-------------------------------"
   done

loss1=$(awk 'NR==60 {print $18}' epochData.txt)
loss2=$(awk 'NR==120 {print $18}' epochData.txt)
loss3=$(awk 'NR==180 {print $18}' epochData.txt)
loss4=$(awk 'NR==240 {print $18}' epochData.txt)
loss5=$(awk 'NR==300 {print $18}' epochData.txt)
loss6=$(awk 'NR==360 {print $18}' epochData.txt)

val_loss1=$(awk 'NR==60 {print $24}' epochData.txt)
val_loss2=$(awk 'NR==120 {print $24}' epochData.txt)
val_loss3=$(awk 'NR==180 {print $24}' epochData.txt)
val_loss4=$(awk 'NR==240 {print $24}' epochData.txt)
val_loss5=$(awk 'NR==300 {print $24}' epochData.txt)
val_loss6=$(awk 'NR==360 {print $24}' epochData.txt)

printf 'Loss Sum = %.3f\n' "$( printf '%f + %f + %f + %f + %f + %f\n' "$loss1" "$loss2" "$loss3" "$loss4" "$loss5" "$loss6" | bc )" >> summaryResults.txt
lossSum=$(awk 'NR==1 {print $4}' summaryResults.txt)
printf '\nAverage Loss: '
awk "BEGIN {print $lossSum/6}"

printf 'Loss_Val Sum = %.3f\n' "$( printf '%f + %f + %f + %f + %f + %f\n' "$val_loss1" "$val_loss2" "$val_loss3" "$val_loss4" "$val_loss5" "$val_loss6" | bc )" >> summaryResults.txt
valLossSum=$(awk 'NR==2 {print $4}' summaryResults.txt)
printf "\nAverage Validation Loss: "
awk "BEGIN {print $valLossSum/6}"


