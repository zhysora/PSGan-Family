############## arguments ##############
model = "ratio"      # model name
dataset = "WV-2"     # dataset name
blk = 64             # patch-size
ref = 0              # ref or no-ref metrics

# for no-ref test we use 'origin dataset'
if ref == 0: 
    dataset = dataset + "_origin"
# because of the mem limit, patch-size = 32 with batch-size = 32
# patch-size = 64 with batch-size = 8
batch_size = {32: 32, 64: 8} 

# config of dataset, you can find it in 'record.txt' under the specific dataset fold
# row and col is to joint patches images to origin size
test_cnt = { 32: {"QB": 486, "GF-2": 3965, "QB_origin": 486, "GF-2_origin": 3965},
             64: {"QB": 117, "GF-2": 960, "QB_origin": 117, "GF-2_origin": 960, "GF-1": 380, "GF-1_origin": 380, "WV-2": 324, "WV-2_origin": 324} }
row = {32: {"QB": 18, "GF-2": 61, "QB_origin": 18, "GF-2_origin": 61},
       64: {"QB": 9, "GF-2": 30, "QB_origin": 9, "GF-2_origin": 30, "GF-1": 19, "GF-1_origin": 19, "WV-2": 18, "WV-2_origin": 18} }
col = {32: {"QB": 27, "GF-2": 65, "QB_origin": 27, "GF-2_origin": 65},
       64: {"QB": 13, "GF-2": 32, "QB_origin": 13, "GF-2_origin": 32, "GF-1": 20, "GF-1_origin": 20, "WV-2": 18, "WV-2_origin": 18} }

############## shell ##############       
import os
os.system('cd eval; nohup python -u eval_one.py \
    --input_dir /data/zh/PSOutput/%s_test_%d_%s/ \
    --num %d \
    --blk %d \
    --row %d \
    --col %d \
    --ref %d \
    > ../model/%s/eval_%s_%d.log 2>&1 & ' % 
    (dataset, blk, model, test_cnt[blk][dataset], blk, row[blk][dataset], col[blk][dataset], ref, 
     model.lower(), dataset, blk) )
