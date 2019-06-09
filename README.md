# MathorNet
This is just a small convolutional neural network designed for the final assignment

![](https://github.com/mathors/MathorNet/blob/master/img/MathorNet2.png?raw=true)

MathorNet is based on LeNet and Imitate Vgg. The accuracy of 84.33 was finally obtained on the cifar-10 test data set. You can refer to Inception V3 or ResNet for retrofit, which may be better

[View Code](https://github.com/mathors/MathorNet/blob/master/MathorNet.ipynb)



If you want to know more detailed parameter Settings, can copy the code below to http://ethereon.github.io/netscope/#/editor

```
name: "MathorNet"
input: "data"
input_dim: 60000
input_dim: 3
input_dim: 32
input_dim: 32
layers {
  bottom: "data"
  top: "conv1_1"
  name: "conv1_1"
  type: CONVOLUTION
  convolution_param {
    num_output: 64
    pad: 1
    kernel_size: 3
  }
}
layers {
  bottom: "conv1_1"
  top: "conv1_1"
  name: "conv1_2"
  type: CONVOLUTION
  convolution_param {
    num_output: 64
    pad: 1
    kernel_size: 3
  }
}
layers {
  bottom: "conv1_1"
  top: "conv1_2"
  name: "pool1"
  type: POOLING
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layers {
  bottom: "conv1_2"
  top: "pool1"
  name: "relu1"
  type: RELU
}
################ 第一部分结束
layers {
  bottom: "pool1"
  top: "conv2_1"
  name: "conv2_1"
  type: CONVOLUTION
  convolution_param {
    num_output: 128
    pad: 1
    kernel_size: 3
  }
}
layers {
  bottom: "conv2_1"
  top: "conv2_1"
  name: "conv2_2"
  type: CONVOLUTION
  convolution_param {
    num_output: 128
    pad: 1
    kernel_size: 3
  }
}
layers {
  bottom: "conv2_1"
  top: "conv2_2"
  name: "pool2"
  type: POOLING
  pooling_param {
    pool: MAX
    kernel_size: 2
    pad: 1
    stride: 2
  }
}
layers {
  bottom: "conv2_2"
  top: "pool2"
  name: "relu2"
  type: RELU
}
################ 第二部分结束
layers {
  bottom: "pool2"
  top: "conv3_1"
  name: "conv3_1"
  type: CONVOLUTION
  convolution_param {
    num_output: 128
    pad: 1
    kernel_size: 3
  }
}
layers {
  bottom: "conv3_1"
  top: "conv3_1"
  name: "conv3_2"
  type: CONVOLUTION
  convolution_param {
    num_output: 128
    pad: 1
    kernel_size: 3
  }
}
layers {
  bottom: "conv3_1"
  top: "conv3_1"
  name: "conv3_3"
  type: CONVOLUTION
  convolution_param {
    num_output: 128
    pad: 1
    kernel_size: 1
  }
}
layers {
  bottom: "conv3_1"
  top: "conv3_3"
  name: "pool3"
  type: POOLING
  pooling_param {
    pool: MAX
    kernel_size: 2
    pad: 1
    stride: 2
  }
}
layers {
  bottom: "conv3_3"
  top: "pool3"
  name: "relu3"
  type: RELU
}
################ 第三部分结束
layers {
  bottom: "pool3"
  top: "conv4_1"
  name: "conv4_1"
  type: CONVOLUTION
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
  }
}
layers {
  bottom: "conv4_1"
  top: "conv4_1"
  name: "conv4_2"
  type: CONVOLUTION
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
  }
}
layers {
  bottom: "conv4_1"
  top: "conv4_1"
  name: "conv4_3"
  type: CONVOLUTION
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 1
  }
}
layers {
  bottom: "conv4_1"
  top: "conv4_3"
  name: "pool4"
  type: POOLING
  pooling_param {
    pool: MAX
    kernel_size: 2
    pad: 1
    stride: 2
  }
}
layers {
  bottom: "conv4_3"
  top: "pool4"
  name: "relu4"
  type: RELU
}
################ 第四部分结束
layers {
  bottom: "pool4"
  top: "conv5_1"
  name: "conv5_1"
  type: CONVOLUTION
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
  }
}
layers {
  bottom: "conv5_1"
  top: "conv5_1"
  name: "conv5_2"
  type: CONVOLUTION
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
  }
}
layers {
  bottom: "conv5_1"
  top: "conv5_1"
  name: "conv5_3"
  type: CONVOLUTION
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 1
  }
}
layers {
  bottom: "conv5_1"
  top: "conv5_3"
  name: "pool5"
  type: POOLING
  pooling_param {
    pool: MAX
    kernel_size: 2
    pad: 1
    stride: 2
  }
}
layers {
  bottom: "conv5_3"
  top: "pool5"
  name: "relu5"
  type: RELU
}
################ 第五部分结束
layers {
  bottom: "pool5"
  top: "fc1"
  name: "fc1"
  type: INNER_PRODUCT
  inner_product_param {
    num_output: 1024
  }
}
layers {
  bottom: "fc1"
  top: "fc1"
  name: "relu1"
  type: RELU
}
layers {
  bottom: "fc1"
  top: "fc1"
  name: "drop1"
  type: DROPOUT
  dropout_param {
    dropout_ratio: 0.5
  }
}
layers {
  bottom: "fc1"
  top: "fc2"
  name: "fc2"
  type: INNER_PRODUCT
  inner_product_param {
    num_output: 1024
  }
}
layers {
  bottom: "fc2"
  top: "fc2"
  name: "relu2"
  type: RELU
}
layers {
  bottom: "fc2"
  top: "fc2"
  name: "drop2"
  type: DROPOUT
  dropout_param {
    dropout_ratio: 0.5
  }
}
layers {
  bottom: "fc2"
  top: "fc3"
  name: "fc3"
  type: INNER_PRODUCT
  inner_product_param {
    num_output: 10
  }
}

################ 全连接结束
```

