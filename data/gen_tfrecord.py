############## import ##############
import os,sys
import tensorflow as tf
import gdal
import numpy as np

############## arguments ##############
satellite = 'WV-2' # name of dateset
blk = 64           # patch size

# config of dataset, you can find it in 'record.txt' under the specific dataset fold
train_cnt ={ 32: {"QB": 53857, "GF-2": 83062},
             64: {"QB": 13460, "GF-2": 20765, "GF-1": 25038, "WV-2": 11552} }
test_cnt = { 32: {"QB": 486, "GF-2": 3965, "QB_origin": 486, "GF-2_origin": 3965},
             64: {"QB": 117, "GF-2": 960, "QB_origin": 117, "GF-2_origin": 960, "GF-1": 380, "GF-1_origin": 380, "WV-2": 324, "WV-2_origin": 324} }

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(inputfiles, name, mode):
    num_examples = len(inputfiles)
    filename = os.path.join(output_dir,name + '.tfrecords')
    print ('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for (file, i) in zip(inputfiles, range(num_examples)):
        print (file, i)
        img_name = '%s_%d' % (mode, i)
        mul_filename = '%s_mul.tif' % file
        blur_filename = '%s_lr.tif' % file
        blur_u_filename = '%s_lr_u.tif' % file
        pan_filename = '%s_pan.tif' % file
        pan_d_filename = '%s_pan_d.tif' % file

        if mode != 'origin_test':
            im_mul_raw = gdal.Open(mul_filename).ReadAsArray().transpose(1, 2, 0).tostring()
        else:
            im_mul_raw = np.zeros([blk * 4, blk * 4, 4], dtype=np.int16).tostring()

        im_blur_raw = gdal.Open(blur_filename).ReadAsArray().transpose(1, 2, 0).tostring()
        im_blur_u_raw = gdal.Open(blur_u_filename).ReadAsArray().transpose(1, 2, 0).tostring()
        im_pan_raw = gdal.Open(pan_filename).ReadAsArray().reshape([blk * 4, blk * 4, 1]).tostring()
        im_pan_d_raw = gdal.Open(pan_d_filename).ReadAsArray().reshape([blk * 4, blk * 4, 1]).tostring()
        
        example = tf.train.Example(features=tf.train.Features(feature={
            'im_name': _bytes_feature(img_name.encode()),
            'im_mul_raw': _bytes_feature(im_mul_raw),
            'im_blur_raw':_bytes_feature(im_blur_raw),
            'im_blur_u_raw':_bytes_feature(im_blur_u_raw),
            'im_pan_raw':_bytes_feature(im_pan_raw),
            'im_pan_d_raw':_bytes_feature(im_pan_d_raw)}))
        writer.write(example.SerializeToString())
    writer.close()

if __name__ == "__main__":
    traincount = train_cnt[blk][satellite]
    testcount = test_cnt[blk][satellite]
    origintestcount = test_cnt[blk][satellite + '_origin']
    data_dir = '/data/zh/PSData/Dataset'
    output_dir = "/data/zh/PSData/TFRecords"

    trainfiles = ['%s/%s/train_%d/%d'% (data_dir, satellite, blk, number) for number in range(traincount)]
    testlist = ['%s/%s/test_%d/%d' % (data_dir, satellite, blk, number) for number in range(testcount)]
    origintestfiles = ['%s/%s/test_origin_%d/%d' % (data_dir, satellite, blk, number) for number in range(origintestcount)]

    convert_to(trainfiles, '%s_train_%d' % (satellite, blk), 'train')
    convert_to(testlist, '%s_test_%d' % (satellite, blk), 'test')
    convert_to(origintestfiles, '%s_origin_test_%d' % (satellite, blk), 'origin_test')

