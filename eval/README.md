# Explanation of The Suffix 

patch_size = 32 for example

| suffix | size | explanation
| - | - | -
｜mul         | (128,128,4)    | the multi-spectral image for ground-truth
｜mul_hat     | (128,128,4)    | the prediction of mul, the fusion image, the ouput of model
｜blur        | (32,32,4)      | the low-resolution of mul, as the input of model 
｜blur_u      | (128,128,4)    | upsample the blur to the same size of mul, as the input of model
｜pan         | (128,128,1)    | the panchromatic image, as the input of model
｜pan_d       | (128,128,1)    | downsample and then upsample the pan, get the blur version of pan
｜pan_d_hat   | (128,128,1)    | the prediction of pan_d, the output of model