# coding=utf-8

############## import ##############
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import math
import time
import collections
import os
import json

import sys
sys.path.append("../../")

from utils import array2raster, load_examples, save_images

############## arguments ##############

parser = argparse.ArgumentParser()
parser.add_argument("--train_tfrecord", help="filename of train_tfrecord",default="/home/zhouhuanyu/data/psgan/train.tfrecords")
parser.add_argument("--test_tfrecord", help="filename of test_tfrecord", default="/home/zhouhuanyu/data/psgan/test.tfrecords")
parser.add_argument("--mode", required=True, choices=["train","test"])
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--checkpoint", default=None, help="directory with checkpoints")
parser.add_argument("--max_steps", type=int, help="max training steps")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=50, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=50, help="write current training images ever display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps")

parser.add_argument("--batch_size",type=int, default=32, help="number of images in batch")

parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")

parser.add_argument("--ndf", type=int, default=32, help="number of generator filters in first conv layer")
parser.add_argument("--train_count", type=int, default=64000,help="number of training data")
parser.add_argument("--test_count", type=int, default=384, help="number of test data")
parser.add_argument("--gpus", help="gpus to run", default="0")
parser.add_argument("--blk", type=int, default=32, help="size of lr image")

a=parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = a.gpus

EPS = 1e-12
Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train")
def conv(batch_input, kernel_size, out_channels, stride):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [kernel_size, kernel_size, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        #padded_input = tf.pad(batch_input,[[0,0],[1,1],[1,1],[0,0]], mode="CONSTANT")
        #conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        conv = tf.nn.conv2d(batch_input, filter, [1,stride, stride, 1], padding='SAME')
        return conv

def lrelu(x, a):
    with tf.name_scope("lrelu"):
        x=tf.identity(x) # 转换成tensor
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def batchnorm(input):
    with tf.variable_scope("batchnorm"):
        input = tf.identity(input)
        channels = input.get_shape()[3]
        offset = tf.get_variable("offset",[channels], dtype=tf.float32, initializer=tf.zeros_initializer()) 
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0,0.02))
        mean, variance = tf.nn.moments(input,axes=[0,1,2], keep_dims=False) # 计算均值和方差
        variance_epsilon= 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized

def strided_conv(batch_input, kernel_size, out_channels):
    with tf.variable_scope("strided_conv"):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [kernel_size, kernel_size, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        strided_conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height*2, in_width*2, out_channels], [1,2,2,1], padding='SAME') # 转置卷积，图像的宽和高都翻倍
        return strided_conv

def strided_conv_2(batch_input, kernel_size, out_channels):
    with tf.variable_scope("strided_conv"):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [kernel_size, kernel_size, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        strided_conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height*4, in_width*4, out_channels], [1,4,4,1], padding='SAME') # 转置卷积，图像的宽和高都翻倍
        return strided_conv

def create_generator(generator_inputs1, generator_inputs2, generator_outputs_channels):
    layers=[] # 特征提取网络
    inputs1_u = []
    with tf.variable_scope("encoder_1_1"): # 分支1 128*128*1             
        output = conv(generator_inputs2, 3, 32, 1) # 3*3*32/1   128*128*32
        layers.append(output)
    with tf.variable_scope("encoder_2_1"):
        rectified = lrelu(layers[-1], 0.2)
        convolved = conv(rectified, 3, 32, 1)      # 3*3*32/1   128*128*32
        layers.append(convolved)
    with tf.variable_scope("encoder_3_1"):
        rectified = lrelu(layers[-1], 0.2)
        convolved = conv(rectified, 2, 64, 2)      # 2*2*64/2   64*64*64   
        layers.append(convolved)

    with tf.variable_scope("encoder_1_2"): # 分支2 多光谱图 32*32*4
        output = conv(generator_inputs1, 3, 32, 1) # 3*3*32/1   32*32*32
        layers.append(output)
    with tf.variable_scope("encoder_2_2"): # 特征层上采样 
        rectified = lrelu(layers[-1], 0.2)
        strided_convolved = strided_conv_2(rectified, 4, 32) # 4*4*32/4     128*128*32
        layers.append(strided_convolved)
        inputs1_u.append(strided_convolved)

    with tf.variable_scope("encoder_3_2"):
        rectified = lrelu(layers[-1], 0.2)
        convolved = conv(rectified, 2, 64, 2)      # 2*2*64/2   64*64*64
        layers.append(convolved)
    # 特征融合网络
    concat1 = tf.concat([layers[-1], layers[-1-3]], 3) # 拼接两个tensor, 在维度3上（channel）64*64*128
    with tf.variable_scope("encoder_4"):
        rectified = lrelu(concat1, 0.2)
        convolved = conv(rectified, 3, 128, 1)     # 3*3*128/1  64*64*128
        layers.append(convolved)
    with tf.variable_scope("encoder_5"):
        rectified = lrelu(layers[-1], 0.2)
        convolved = conv(rectified, 3, 128, 1)     # 3*3*128/1  64*64*128
        layers.append(convolved)
    with tf.variable_scope("encoder_6"):
        rectified = lrelu(layers[-1], 0.2)
        convolved = conv(rectified, 3, 256, 2)     # 3*3*256/2  32*32*256
        layers.append(convolved)
    # 图像重建网络
    with tf.variable_scope("decoder_7"):
        rectified = lrelu(layers[-1], 0.2)
        convolved = conv(rectified, 1, 256, 1)     # 1*1*256/1  32*32*256
        layers.append(convolved)

    with tf.variable_scope("decoder_8"):
        rectified = lrelu(layers[-1], 0.2)
        convolved = conv(rectified, 3, 256, 1)     # 3*3*256/1  32*32*256
        layers.append(convolved)

    with tf.variable_scope("decoder_9"):
        rectified = lrelu(layers[-1], 0.2)
        strided_convolved = strided_conv(rectified, 2, 128) # 2*2*128/2     64*64*128
        layers.append(strided_convolved)

    concat2 = tf.concat([layers[-1], layers[-1-4]], 3)      # 64*64*256

    with tf.variable_scope("decoder_10"):
        rectified = lrelu(concat2, 0.2)
        convolved = conv(rectified, 3, 128, 1)              # 3*3*128/1     64*64*128
        layers.append(convolved)

    with tf.variable_scope("decoder_11"):
        rectified = lrelu(layers[-1], 0.2)
        strided_convolved = strided_conv(rectified, 2, 128) # 2*2*128/2     128*128*128
        layers.append(strided_convolved)

    concat3 = tf.concat([layers[-1], layers[-1-9], layers[-1-12]], 3) # 128*128*192

    with tf.variable_scope("decoder_12"):
        rectified = lrelu(concat3, 0.2)
        convolved = conv(rectified, 3, 64, 1)               # 3*3*64/1  128*128*64
        layers.append(convolved)

    with tf.variable_scope("decoder_13"):
        rectified = lrelu(layers[-1], 0.2)
        convolved = conv(rectified, 3, generator_outputs_channels, 1) # 3*3*4/1 128*128*4
        output = tf.nn.relu(convolved)
        layers.append(output)

    return layers[-1], inputs1_u[0]

def create_model(inputs1, inputs2, targets, inputs3):
    def create_discriminator(discrim_inputs, discrim_targets):
        n_layers = 3
        layers = []

        input = tf.concat([discrim_inputs, discrim_targets], 3) # 128*128*8

        with tf.variable_scope("layer_1"):
            convolved = conv(input, 3, a.ndf, 2) # 3*3*32/2     64*64*32  
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = a.ndf * min(2 ** (i + 1), 8) # 输出通道数每次*2，最多到8
                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1，除最后一层，每层宽高减半
                convolved = conv(layers[-1], 3, out_channels, stride=stride)
                rectified = lrelu(convolved, 0.2)
                layers.append(rectified)
        # 16*16*256
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = conv(rectified, 3, 1, 1) # 3*3*1/1  16*16*1  
            output = tf.sigmoid(convolved)
            layers.append(output)

        return layers[-1]

    with tf.variable_scope("generator") as scope:
        out_channels = int(targets.get_shape()[-1])
        outputs, inputs1_u = create_generator(inputs1, inputs2, out_channels)

    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            predict_real = create_discriminator(inputs3, targets)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            predict_fake = create_discriminator(inputs3, outputs)

    with tf.name_scope("discriminator_loss"):
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

    with tf.name_scope("generator_loss"):
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake+EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        predict_real = predict_real,
        predict_fake = predict_fake,
        discrim_loss = ema.average(discrim_loss),
        discrim_grads_and_vars = discrim_grads_and_vars,
        gen_loss_GAN = ema.average(gen_loss_GAN),
        gen_loss_L1 = ema.average(gen_loss_L1),
        gen_grads_and_vars = gen_grads_and_vars,
        outputs= outputs,
        train = tf.group(update_losses, incr_global_step, gen_train),
    )

def prework():
    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.mode == "test":
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")

    for k,v in a._get_kwargs():
        print (k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

def main():
    prework()

    # load data
    examples = load_examples(a)

    # load model
    model = create_model(examples.blur, examples.pan, examples.mul, examples.blur_u)

    with tf.name_scope("images"):
        display_fetches = {
            "imnames" : examples.imnames,
            "blur_u"  : examples.blur_u,
            "blur"    : examples.blur,
            "pan"     : examples.pan,
            "mul"     : examples.mul,
            "mul_hat" : model.outputs,
        }

    with tf.name_scope("blur_u_summary"):
        tf.summary.image("blur_u", examples.blur_u)

    with tf.name_scope("blur_summary"):
        tf.summary.image("blur", examples.blur)

    with tf.name_scope("mul_summary"):
        tf.summary.image("mul", examples.mul)

    with tf.name_scope("mul_hat_summary"):
        tf.summary.image("mul_hat", model.outputs)

    with tf.name_scope("predict_real_summary"):
        tf.summary.image("predict_real", model.predict_real)

    with tf.name_scope("predict_fake_summary"):
        tf.summary.image("predict_fake", model.predict_fake)

    tf.summary.scalar("discriminator_loss", model.discrim_loss)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
        tf.summary.histogram(var.op.name + "/gradients", grad)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq >0 ) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session()  as sess:
        print("parameter_count = ", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 100
        if a.max_epochs is not None:
            max_steps = examples.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        if a.mode == "test":
            max_steps = min(examples.steps_per_epoch, max_steps)
            for step in range(max_steps):
                results = sess.run(display_fetches)
                save_images(results, a)
            print ("test over")
        else:
            start = time.time()
            sv_gloss = -1

            for step  in range(max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step ==max_steps - 1)

                options = None
                run_metadata = None
                if should(a.trace_freq):
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(a.progress_freq) or should(a.save_freq):
                    fetches["discrim_loss"] = model.discrim_loss
                    fetches["gen_loss_GAN"] = model.gen_loss_GAN
                    fetches["gen_loss_L1"] = model.gen_loss_L1

                if should(a.summary_freq):
                    fetches["summary"] = sv.summary_op

                if should(a.display_freq):
                    fetches["display"] = display_fetches
                
                results = sess.run(fetches, options=options, run_metadata=run_metadata)

                if should(a.summary_freq):
                    print("recording summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if should(a.display_freq):
                    print("saving display images")
                    save_images(results["display"], a, step=results["global_step"])

                if should(a.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(a.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                    train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                    rate = (step + 1) * a.batch_size / (time.time() - start)
                    remaining = (max_steps - step) * a.batch_size / rate
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (
                    train_epoch, train_step, rate, remaining / 60))
                    print("discrim_loss", results["discrim_loss"])
                    print("gen_loss_GAN", results["gen_loss_GAN"])
                    print("gen_loss_L1", results["gen_loss_L1"])

                if should(a.save_freq):
                    print("saving model")
                    gloss = results["gen_loss_GAN"] * a.gan_weight + results["gen_loss_L1"] * a.l1_weight
                    if sv_gloss == -1 or gloss < sv_gloss or True:
                        sv_gloss = gloss
                        saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)
                    else:
                        print("skipped because of worse loss")
                
                if sv.should_stop():
                    break

if __name__ == "__main__":
    main()
