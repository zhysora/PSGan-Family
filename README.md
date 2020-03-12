# How to Use this Project

This project is an experiment comparing the performance among the various state-of-art models in Pan-Sharpening filed. To utilize the codes, you can follow these steps:

1. build your own dataset
2. train and test the models 
3. evaluate the results

# Build Your Own Dataset

This project is based on Tensorflow, so the dataset will finally be made into '.tfrecords' format. The related codes are all in 'data' folder. 

Before running the codes, you must fix the path in codes according to your images. You can refer to our structure of the dataset. You can see detail in 'data' folder.

The order to run the codes:
```
cd PROJECT_PATH
cd data
python handle_raw.py
python gen_dataset.py
python gen_tfrecord.py
```
# Train and Test the Models 

You can just modify the args in run.py and run
```
cd PROJECT_PATH
python run.py
```

Generally the logs will be generated below the specific model folder.

Or you can directly run the specific model code for a more detail using. You can see the help of the args in each model codes. For example
```
cd PROJECT_PATH
cd model/psgan
nohup python -u psgan.py --mode train --train_tfrecord ** --test_tfreord ** ... > **.log 2>&1 &
```

# Evaluation the Results

You can just modify the args in eval.py and run
```
cd PROJECT_PATH
python eval.py
```

Generally the logs will be generated below the specific model folder.

Or you can directly run 'eval/eval_one.py' for a more detail using. For example
```
cd PROJECT_PATH
cd eval; nohup python -u eval_one.py --input_dir ** --num ** --blk ** --row ** --col ** --ref ** > **.log 2>&1 & 
```

# Model List

This project implement various sate-of-art Pan-Sharpening models in Tensorflow. The related codes are below 'model' folder, and each model one folder.

## psgan
psgan: psgan orign   

fu-psgan: upsample the MS images using strided convolution 

st-psgan: concatenate the PAN and MS input without two-stream architecture

# Dataset List

We create many datasets for the experiment.

QB_32: data from quickbird satellite, MS images cut into (32, 32, 4), PAN images cut into (128, 128)

QB_64: data from quickbird satellite, MS images cut into (64, 64, 4), PAN images cut into (256, 256)

GF-2_32: data from GaoFen-2 satellite, MS images cut into (32, 32, 4), PAN images cut into (128, 128)

GF-2_64: data from GaoFen-2 satellite, MS images cut into (64, 64, 4), PAN images cut into (256, 256)

GF-1_64: data from GaoFen-1 satellite, MS images cut into (64, 64, 4), PAN images cut into (256, 256)

WV-2_64: data from WorldView-2 satellite, MS images cut into (64, 64, 4), PAN images cut into (256, 256)