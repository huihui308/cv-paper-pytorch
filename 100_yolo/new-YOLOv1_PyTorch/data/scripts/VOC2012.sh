#!/bin/bash
# Ellis Brown
#
# ./VOC2012.sh /home/david/dataset/detect/VOC/VOC2012
#

start=`date +%s`

# handle optional download dir
if [ -z "$1" ]
  then
    # navigate to ~/data
    echo "navigating to ~/data/ ..." 
    mkdir -p ~/data
    cd ~/data/
  else
    # check if is valid directory
    if [ ! -d $1 ]; then
        echo $1 "is not a valid directory"
        exit 0
    fi
    echo "navigating to" $1 "..."
    cd $1
fi

train_file="./VOCtrainval_11-May-2012.tar"
if [ ! -f "${train_file}" ]; then
  echo "Downloading VOC2012 trainval ..."
  # Download the data.
  curl -LO http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
  echo "Done downloading."
else
  echo "Train file already exist"
fi

# Extract data
echo "Extracting trainval ..."
tar -xvf VOCtrainval_11-May-2012.tar
echo "removing tar ..."
#rm VOCtrainval_11-May-2012.tar

end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"