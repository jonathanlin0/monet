apt-get -y install python3-pip
sudo apt-get -y install vim
sudo apt-get install unzip

# yes | pip3 install cython
sudo apt-get -y install python3-dev

pip3 install -r requirements.txt

# cv2 dependencies
apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

chmod +rwx ./train.sh

export KAGGLE_USERNAME=asdfasdfaasdfasf
export KAGGLE_KEY=b09080a36c090e15f19841a4de125b32

# download the datasets
kaggle datasets download trolukovich/food11-image-dataset
unzip food11-image-dataset.zip -d model/datasets
mv model/datasets/food11-image-dataset/ model/datasets/food11

kaggle datasets download -d kmader/food41
unzip food41.zip -d model/datasets
mv model/datasets/food41/ model/datasets/food101