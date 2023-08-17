#!/bin/bash
# Ellis Brown
#
# ./VOC2007.sh /home/david/dataset/detect/VOC/VOC2007
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

# prepare files
train_file="./VOCtrainval_06-Nov-2007.tar"
if [ ! -f "${train_file}" ]; then
  echo "Downloading VOC2007 trainval ..."
  curl -LO http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
  echo "Done downloading train file."
else
  echo "Train file already exist"
fi

test_file="./VOCtest_06-Nov-2007.tar"
if [ ! -f "${test_file}" ]; then
  echo "Downloading VOC2007 test data ..."
  curl -LO http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
  echo "Done test train file."
else
  echo "Test file already exist"
fi

# Extract data
echo "Extracting trainval ..."
tar -xvf VOCtrainval_06-Nov-2007.tar
echo "Extracting test ..."
tar -xvf VOCtest_06-Nov-2007.tar
echo "removing tars ..."
#rm VOCtrainval_06-Nov-2007.tar
#rm VOCtest_06-Nov-2007.tar

end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"