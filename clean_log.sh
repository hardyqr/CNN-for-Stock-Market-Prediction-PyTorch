#!/bin/bash

if [ -d "./logs" ]; then
	rm -r logs
fi

if [ -e "./training_loss_records.csv" ]; then
	rm training_loss_records.csv
fi

if [ $# -eq "1" ]; then
	git add .
	git commit -m $1+" version"
	git push
fi
