# coding: utf-8
import numpy as np


# This function can only used in the convolution
# with out dilate operate.
def naive_convolutional(input, kernel, pad, stride):

    batch_size, in_channels, in_height, in_width = input.shape
    out_channels, in_channels, ksize, _ = kernel.shape

    out_height = int(((in_height - ksize) + 2 * pad) / stride + 1)
    out_width = int(((in_width - ksize) + 2 * pad) / stride + 1)

    # Pad the input with 0
    padded_input = np.pad(input, \
        ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=0)

    # Transform the featuremap and kernel to matrix,
    # prepared for GEMM
    input_shape = (batch_size, in_channels, out_height * out_width,
                    ksize * ksize)
    output_shape = (batch_size, out_channels, out_height * out_width, 1)
    kernel_shape = (out_channels, in_channels, ksize * ksize, 1)

    reshaped_input = np.zeros(input_shape)
    reshaped_output = np.zeros(output_shape)
    reshaped_kernel = kernel.reshape(kernel_shape)

    
    for j in range(out_height):
        row = j * stride
        for k in range(out_width):
            column = k * stride
            for i in range(in_channels):
                for b in range(batch_size):
                    reshaped_input[b, i, j * out_width + k, :] = \
                        padded_input[b, i, row: row + ksize, column: column + ksize] \
                            .reshape((1, -1))
                for l in range(out_channels):
                    for b in range(batch_size):
                        reshaped_output[b, l, j * out_width + k, 0] += \
                            reshaped_input[b, i, j * out_width + k, :].dot(
                                reshaped_kernel[l, i, :, :]
                            )
    
    output = reshaped_output.reshape((batch_size, out_channels, out_height, out_width))

    return output, reshaped_input, reshaped_output


def naive_convolutional_deriviation(gradient, input, kernel,
                                    reshaped_input, reshaped_output,
                                    pad, stride):

    out_channels, in_channels, ksize, _ = kernel.shape
    batch_size, _, in_height, in_width = input.shape

    out_height = int(((in_height - ksize) + 2 * pad) / stride + 1)
    out_width = int(((in_width - ksize) + 2 * pad) / stride + 1)

    reshaped_kernel_shape = (out_channels, in_channels, ksize * ksize, 1)

    # Construct the gradient
    reshaped_output_gradient = gradient.reshape(reshaped_output.shape)
    reshaped_kernel = kernel.reshape(reshaped_kernel_shape)

    kernel_gradient = np.zeros(kernel.shape)
    reshaped_input_gradient = np.zeros(reshaped_input.shape)
    input_gradient = np.zeros(
        (batch_size, in_channels, in_height + pad * 2, in_width + pad * 2)
    )

    # Compute the gradient of kernel
    for b in range(batch_size):
        for o in range(out_channels):
            for i in range(in_channels):
                kernel_gradient[o, i, :, :] += \
                    reshaped_input[b, i, :, :].T.dot(
                        reshaped_output_gradient[b, o, :, :]
                    ).reshape((ksize, ksize))
    
    # Compute the gradient of input
    for b in range(batch_size):
        for o in range(out_channels):
            for i in range(in_channels):
                reshaped_input_gradient[b, i] += \
                    reshaped_output_gradient[b, o, :, :].dot(
                        reshaped_kernel[o, i, :, :].T
                    )
    
    # Gather the gradient of input
    for j in range(out_height):
        row = j * stride
        for k in range(out_width):
            column = k * stride
            input_gradient[:, :, row: row + ksize, column: column + ksize] += \
                reshaped_input_gradient[:, :, j * out_width + k, :].reshape((batch_size, in_channels, ksize, ksize))
    
    input_gradient = input_gradient[:, :, pad: in_height + pad, pad: in_width + pad]
    
    return kernel_gradient, input_gradient


def maxpool(input, stride=2):
    
    batch_size, in_channels, in_height, in_width = input.shape
    out_height, out_width = int(in_height / stride), int(in_width / stride)

    mask = np.zeros(input.shape)
    output = np.zeros((batch_size, in_channels, out_height, out_width))

    
    for h in range(out_height):
        row = h * stride
        for w in range(out_width):
            column = w * stride
            for b in range(batch_size):
                for i in range(in_channels):
                    index = np.argmax(input[b, i, row: row + stride, column: column + stride])
                    index = (int(index/stride), index % stride)
                    mask[b, i, row + index[0], column + index[1]] = 1
                    output[b, i, h, w] = input[b, i, row + index[0], column + index[1]]

    return mask, output


def maxpool_deriviation(gradient, mask, stride):

    batch_size, in_channels, out_height, out_width = gradient.shape
    dilated_gradient = np.zeros(mask.shape)

    
    for h in range(out_height):
        row = stride * h
        for w in range(out_width):
            column = stride * w
            for b in range(batch_size):
                for i in range(in_channels):
                    dilated_gradient[b, i, row: row + stride, column: column + stride] \
                        = gradient[b, i, h, w]

    return dilated_gradient * mask


def conv_initializer(kernel_shape):
    return np.random.random(kernel_shape) / 100.0


def naive_initializer(weights_shape):
    return np.random.random(weights_shape) / 100.0 + 0.001
