# The Structure Of Our Dataset (GF-1 for example)

```
DataDir
├── Raw
│   ├── GF-1
│       ├── 1-MUL.TIF
│       ├── 1-PAN.TIF
│       ├── 2-MUL.TIF
│       ├── 2-PAN.TIF
│       ├── 3-MUL.TIF
│       ├── 3-PAN.TIF
│       ├── 4-MUL.TIF
│       ├── 4-PAN.TIF
│       ├── 5-MUL.TIF
│       ├── 5-PAN.TIF
│       ├── 6-MUL.TIF
│       ├── 6-PAN.TIF
│       ├── 7-MUL.TIF
│       ├── 7-PAN.TIF
│       ├── 8-MUL.TIF
│       ├── 8-PAN.TIF
│       ├── 9-MUL.TIF
│       └── 9-PAN.TIF
│  
├── Dataset
│   ├── GF-1
│       ├── 1_lr.tif
│       ├── 1_lr_u.tif
│       ├── 1_mul.tif
│       ├── 1_pan_d.tif
│       ├── 1_pan.tif
│       ├── 2_lr.tif
│       ├── 2_lr_u.tif
│       ├── 2_mul.tif
│       ├── 2_pan_d.tif
│       ├── 2_pan.tif
│       ├── 3_lr.tif
│       ├── 3_lr_u.tif
│       ├── 3_mul.tif
│       ├── 3_pan_d.tif
│       ├── 3_pan.tif
│       ├── 4_lr.tif
│       ├── 4_lr_u.tif
│       ├── 4_mul.tif
│       ├── 4_pan_d.tif
│       ├── 4_pan.tif
│       ├── 5_lr.tif
│       ├── 5_lr_u.tif
│       ├── 5_mul.tif
│       ├── 5_pan_d.tif
│       ├── 5_pan.tif
│       ├── 6_lr.tif
│       ├── 6_lr_u.tif
│       ├── 6_mul.tif
│       ├── 6_pan_d.tif
│       ├── 6_pan.tif
│       ├── 7_lr.tif
│       ├── 7_lr_u.tif
│       ├── 7_mul.tif
│       ├── 7_pan_d.tif
│       ├── 7_pan.tif
│       ├── 8_lr.tif
│       ├── 8_lr_u.tif
│       ├── 8_mul.tif
│       ├── 8_pan_d.tif
│       ├── 8_pan.tif
│       ├── 9_lr.tif
│       ├── 9_lr_u.tif
│       ├── 9_mul_o.tif
│       ├── 9_mul_o_u.tif
│       ├── 9_mul.tif
│       ├── 9_pan_d.tif
│       ├── 9_pan_o_d.tif
│       ├── 9_pan_o.tif
│       ├── 9_pan.tif
│       ├── test_64
│       ├── test_origin_64
│       └── train_64
│   
└── TFRecords
    ├── GF-1_origin_test_64.tfrecords
    ├── GF-1_test_64.tfrecords
    └── GF-1_train_64.tfrecords
```

There are 9 pairs of raw images in GF-1 Dataset. The raw images are in 'DataDir/Raw/GF-1', named '1-MUL.TIF, 1-PAN.TIF, 2-MUL.TIF ...'

handle_raw.py will input the raw images and then output the images after preprocessing in 'DataDir/Dataset/GF-1', named '1_lr.tif, 1_lr_u.tif, 1_mul.tif ...'

gen_dataset.py will input the outputs of handle_raw.py, and crop the images to patches. 'DataDir/Dataset/GF-1/train_64' is cropped by random sampling for training, 'DataDir/Dataset/GF-1/test_64' is cropped in oder with some overlapping for testing to caculate the reference metircs, 'DataDir/Dataset/GF-1/test_origin_64' is same as the 'DataDir/Dataset/GF-1/test_64' but without ground-truth mul image as target, and is to caculate the no-reference metircs. 

gen_tfrecord.py will output the final data format  '.tfrecords' in 'DataDir/GF-1/TFRecords'

So if you want to use our codes to build your own dataset just put the raw images into 'DataDir/Raw/**',  numbered as '1-MUL.TIF, 1-PAN.TIF ...' . Then run handle_raw.py, gen_dataset.py, gen_tfrecords.py, you can get the result in 'DataDir/TFrecords/'. 


