import gdal,osr
import tensorflow as tf
import collections
import os
import numpy as np

def array2raster(newRasterfn, rasterOrigin, pixelWidth, pixelHeight, array, bandSize):
    rasterOrigin = (-123.25745,45.43013)
    pixelWidth = 2.4
    pixelHeight = 2.4
    
    if (bandSize == 4):
        cols = array.shape[2]
        rows = array.shape[1]
        originX = rasterOrigin[0]
        originY = rasterOrigin[1]

        driver = gdal.GetDriverByName('GTiff')

        outRaster = driver.Create(newRasterfn, cols, rows, 4, gdal.GDT_UInt16)
        outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
        for i in range(1, 5):
            outband = outRaster.GetRasterBand(i)
            outband.WriteArray(array[i - 1, :, :])
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromEPSG(4326)
        outRaster.SetProjection(outRasterSRS.ExportToWkt())
        outband.FlushCache()
    elif (bandSize == 1):
        cols = array.shape[1]
        rows = array.shape[0]
        originX = rasterOrigin[0]
        originY = rasterOrigin[1]

        driver = gdal.GetDriverByName('GTiff')

        outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_UInt16)
        outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))

        outband = outRaster.GetRasterBand(1)
        outband.WriteArray(array[:, :])

def load_examples(args):
    if args.mode == 'train':
        filename_queue = tf.train.string_input_producer([args.train_tfrecord])
    elif args.mode =='test':
        filename_queue = tf.train.string_input_producer([args.test_tfrecord])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'im_name': tf.FixedLenFeature([],tf.string),
                                           'im_mul_raw': tf.FixedLenFeature([], tf.string),
                                           'im_blur_raw': tf.FixedLenFeature([], tf.string),
                                           'im_blur_u_raw': tf.FixedLenFeature([], tf.string),
                                           'im_pan_raw': tf.FixedLenFeature([], tf.string),
                                           'im_pan_d_raw': tf.FixedLenFeature([], tf.string)
                                       })

    im_mul_raw = tf.decode_raw(features['im_mul_raw'], tf.int16)
    im_mul_raw = tf.reshape(im_mul_raw, [args.blk*4, args.blk*4, 4])
    im_mul_raw = tf.cast(im_mul_raw,tf.float32)
    im_blur_raw = tf.decode_raw(features['im_blur_raw'], tf.int16)
    im_blur_raw = tf.reshape(im_blur_raw, [args.blk, args.blk, 4])
    im_blur_raw=tf.cast(im_blur_raw, tf.float32)
    im_blur_u_raw = tf.decode_raw(features['im_blur_u_raw'], tf.int16)
    im_blur_u_raw = tf.reshape(im_blur_u_raw, [args.blk*4, args.blk*4, 4])
    im_blur_u_raw = tf.cast(im_blur_u_raw, tf.float32)
    im_pan_raw = tf.decode_raw(features['im_pan_raw'], tf.int16)
    im_pan_raw = tf.reshape(im_pan_raw, [args.blk*4, args.blk*4, 1])
    im_pan_raw = tf.cast(im_pan_raw, tf.float32)
    im_pan_d_raw = tf.decode_raw(features['im_pan_d_raw'], tf.int16)
    im_pan_d_raw = tf.reshape(im_pan_d_raw, [args.blk*4, args.blk*4, 1])
    im_pan_d_raw = tf.cast(im_pan_d_raw, tf.float32)

    if args.mode == 'train':
        imnames_batch, blur_batch, pan_batch, blur_u_batch, pan_d_batch, mul_batch = \
            tf.train.shuffle_batch(
            [features['im_name'], im_blur_raw, im_pan_raw, im_blur_u_raw, im_pan_d_raw, im_mul_raw],
            batch_size=args.batch_size, capacity=200, min_after_dequeue=100)
        steps_per_epoch = args.train_count // args.batch_size + (args.train_count % args.batch_size != 0)
    elif args.mode =='test':
        imnames_batch, blur_batch, pan_batch, blur_u_batch, pan_d_batch, mul_batch = \
            tf.train.batch(
            [features['im_name'],im_blur_raw, im_pan_raw, im_blur_u_raw, im_pan_d_raw, im_mul_raw],
            batch_size=args.batch_size, capacity=200)
        steps_per_epoch = args.test_count // args.batch_size + (args.test_count % args.batch_size != 0)

    Examples = collections.namedtuple("Examples", 
        "imnames, blur, pan, blur_u, pan_d, mul, steps_per_epoch")

    return Examples(
        imnames         = imnames_batch,
        blur            = blur_batch,   # blur   32*32*4
        pan             = pan_batch,    # pan    128*128*1
        blur_u          = blur_u_batch, # blur_u 128*128*4
        pan_d           = pan_d_batch,  # pan_d  128*128*1
        mul             = mul_batch,    # mul    128*128*4
        steps_per_epoch = steps_per_epoch,
    )

def save_images(fetches, args, step=None):
    image_dir = os.path.join(args.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    for i, in_path in enumerate(fetches["imnames"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        print (fetches['mul'][i][33][33], ':', fetches['mul_hat'][i][33][33])
        for kind in fetches:
            filename = name + "-" + kind + ".tif"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]

            if kind in ["blur"]:
                array2raster(out_path, [0, 0], 4, 4, contents.transpose(2,0,1), 4)
            elif kind in ["pan", "pan_d", "pan_d_hat"]:
                array2raster(out_path, [0, 0], 1, 1, contents.reshape((args.blk*4,args.blk*4)), 1)
            elif kind in ["blur_u", "mul", "mul_hat"]:
                array2raster(out_path, [0, 0], 1, 1, contents.transpose(2,0,1), 4) 

def trim_image(image, L = 0, R = 2**11):
    return tf.minimum(tf.maximum(image, L), R)

def arrayToHist(grayArray, nums):
    if(len(grayArray.shape) != 2):
        print("length error")
        return None

    w, h = grayArray.shape
    hist = {}
    for k in range(nums):
        hist[k] = 0
    for i in range(w):
        for j in range(h):
            if(hist.get(grayArray[i][j]) is None):
                hist[grayArray[i][j]] = 0
            hist[grayArray[i][j]] += 1
    # normalize
    n = w * h
    for key in hist.keys():
        hist[key] = float(hist[key]) / n
    return hist

def histMatch(grayArray, grayArray_d, nums):
    h_d = arrayToHist(grayArray_d, nums)
    tmp = 0.0
    h_acc = h_d.copy()
    for i in range(nums):
        tmp += h_d[i]
        h_acc[i] = tmp

    h1 = arrayToHist(grayArray, nums)
    tmp = 0.0
    h1_acc = h1.copy()
    for i in range(nums):
        tmp += h1[i]
        h1_acc[i] = tmp
    
    M = np.zeros(nums)

    j = 0
    for i in range(nums):
        
        while j < nums and h1_acc[i] > h_acc[j]:
            j = j + 1
        
        if j == 0:
            M[i] = 0
        else:
            M[i] = j if np.fabs(h_acc[j] - h1_acc[i]) < np.fabs(h_acc[j - 1] - h1_acc[i]) else j - 1
            
    des = M[grayArray]
    return des

def imagesHistMatch(x, y, nums):
    print("run imagesHistMatch")
    x = np.array(x, dtype=np.int32)
    y = np.array(y, dtype=np.int32)

    B, H, W, C = x.shape
    for b in range(B):
        for c in range(C):
            print(b, c)
            x[b, :, :, c] = histMatch(x[b, :, :, c], y[b, :, :, c], nums)
    
    return x
    
