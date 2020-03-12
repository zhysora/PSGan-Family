############## arguments ##############
model = "psgan" # model name
mode = "test"          # mode
dataset = "WV-2"       # dateset name
gpus = "5"             # gpu ids
blk = 64               # patch-size
ref = 0                # ref or no-ref test

# for no-ref test we use 'origin dataset'
if ref == 0: 
    dataset = dataset + "_origin"
# because of the mem limit, patch-size = 32 with batch-size = 32
# patch-size = 64 with batch-size = 8
batch_size = {32: 32, 64: 8} 

# config of dataset, you can find it in 'record.txt' under the specific dataset fold
train_cnt ={ 32: {"QB": 53857, "GF-2": 83062, "QB_origin":0, "GF-2_origin": 0},
             64: {"QB": 13460, "GF-2": 20765, "WV-2": 11552, "GF-1": 25038, "QB_origin":0, "GF-2_origin": 0, "GF-1_origin": 0, "WV-2_origin": 0} }
test_cnt = { 32: {"QB": 486, "GF-2": 3965, "QB_origin": 486, "GF-2_origin": 3965},
             64: {"QB": 117, "GF-2": 960, "QB_origin": 117, "GF-2_origin": 960, "GF-1": 380, "GF-1_origin": 380, "WV-2": 324, "WV-2_origin": 324} }

############## shell ##############
import os
if mode == 'train':
    os.system('cd model/%s; nohup python -u %s.py \
    --mode train \
        --train_tfrecord /data/zh/PSData/TFRecords/%s_train_%d.tfrecords \
        --test_tfrecord /data/zh/PSData/TFRecords/%s_test_%d.tfrecords \
        --output_dir /data/zh/PSOutput/%s_train_%d_%s \
        --summary_freq 0 \
        --progress_freq 200 \
        --trace_freq 0 \
        --display_freq 0 \
        --save_freq 1000 \
        --max_epochs 50 \
        --train_count %d \
        --test_count %d \
        --gpus %s \
        --blk %d \
        --batch_size %d \
        > %s_%s_%d.log 2>&1 &' % 
        (model.lower(), model.lower(), dataset, blk, dataset, blk, dataset, blk, model,
        train_cnt[blk][dataset], test_cnt[blk][dataset], gpus, blk, batch_size[blk],
        mode, dataset, blk) )
else:
    os.system('cd model/%s; nohup python -u %s.py \
        --mode test \
        --train_tfrecord /data/zh/PSData/TFRecords/%s_train_%d.tfrecords \
        --test_tfrecord /data/zh/PSData/TFRecords/%s_test_%d.tfrecords \
        --output_dir /data/zh/PSOutput/%s_test_%d_%s \
        --summary_freq 0 \
        --progress_freq 200 \
        --trace_freq 0 \
        --display_freq 0 \
        --save_freq 1000 \
        --max_epochs 50 \
        --train_count %d \
        --test_count %d \
        --gpus %s \
        --blk %d \
        --batch_size %d \
        --checkpoint /data/zh/PSOutput/%s_train_%d_%s\
        > %s_%s_%d.log 2>&1 &' % 
        (model.lower(), model.lower(), dataset, blk, dataset, blk, dataset, blk, model,
        train_cnt[blk][dataset], test_cnt[blk][dataset], gpus, blk, batch_size[blk],
        dataset.split('_')[0], blk, model, mode, dataset, blk) )  
