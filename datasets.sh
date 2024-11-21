# New datasets added afterwards
cd /scratch/workspace/asureddy_umass_edu-llm_alignment/dataset

# creating captioning dataset folder
[ ! -d "./captioning" ] && mkdir -p "./captioning"

cd ./captioning

# downloading flickr-8k datasets
wget "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr8k.zip"
unzip flickr8k.zip -d ./flickr8k
rm flickr8k.zip
echo "Downloaded Flickr8k dataset successfully."

# VQA
# textvqa dataset
cd ./vqa
mkdir textvqa && cd textvqa
wget https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_train.json
wget https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json
wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
unzip train_val_images.zip -d ./train_val_images
rm train_val_images.zip
echo "Downloaded textVQA train_val dataset successfully."

wget https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_test.json
wget https://dl.fbaipublicfiles.com/textvqa/images/test_images.zip
unzip test_images.zip -d ./test_images
rm test_images.zip
echo "Downloaded textVQA test dataset successfully."
