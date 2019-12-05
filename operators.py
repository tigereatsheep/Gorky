# coding: utf-8
import numpy as np


# This function can only used in the convolution
# with out dilate operate.
def naive_convolutional(input, kernel, pad, stride):

    in_channels, in_height, in_width = input.shape
    out_channels, in_channels, ksize, _ = kernel.shape

    out_height = int(((in_height - ksize) + 2 * pad) / stride + 1)
    out_width = int(((in_width - ksize) + 2 * pad) / stride + 1)

    # Pad the input with 0
    padded_input = np.pad(input, ((0, 0), (pad, pad), (pad, pad)), \
                            'constant', constant_values=0)

    # Transform the featuremap and kernel to matrix,
    # prepared for GEMM
    input_shape = (out_channels, out_height * out_width,
                    ksize * ksize)
    output_shape = (out_channels, out_height * out_width, 1)
    kernel_shape = (out_channels, in_channels, ksize * ksize, 1)

    reshaped_input = np.zeros(input_shape)
    reshaped_output = np.zeros(output_shape)
    reshaped_kernel = kernel.reshape(kernel_shape)

    for i in range(in_channels):
        for j in range(out_height):
            for k in range(out_width):
                row = j * stride
                column = k * stride
                reshaped_input[i, j * out_width + k, :] = \
                    padded_input[i, row: row + ksize, column: column + ksize] \
                        .reshape((1, -1))
                for l in range(out_channels):
                    reshaped_output[l, j * out_width + k, 0] += \
                        reshaped_input[i, j * out_width + k, :].dot(
                            reshaped_kernel[l, i, :, :]
                        )
    
    output = reshaped_output.reshape((out_channels, out_height, out_width))

    return output, reshaped_input, reshaped_output


def naive_convolutional_deriviation(gradient, input, kernel,
                                    reshaped_input, reshaped_output,
                                    pad, stride):

    out_channels, in_channels, ksize, _ = kernel.shape
    _, in_height, in_width = input.shape

    out_height = int(((in_height - ksize) + 2 * pad) / stride + 1)
    out_width = int(((in_width - ksize) + 2 * pad) / stride + 1)

    reshaped_kernel_shape = (out_channels, in_channels, ksize * ksize, 1)

    # Construct the gradient
    reshaped_output_gradient = gradient.reshape(reshaped_output.shape)
    reshaped_kernel = kernel.reshape(reshaped_kernel_shape)

    kernel_gradient = np.zeros(kernel.shape)
    reshaped_input_gradient = np.zeros(reshaped_input.shape)
    input_gradient = np.zeros((in_channels, in_height + pad * 2, in_width + pad * 2))

    # Compute the gradient of kernel
    for i in range(out_channels):
        for j in range(in_channels):
            kernel_gradient[i, j, :, :] = \
                reshaped_input[j, :, :].T.dot(
                    reshaped_output_gradient[i, :, :]
                ).reshape((ksize, ksize))
    
    # Compute the gradient of input
    for i in range(out_channels):
        for j in range(in_channels):
            reshaped_input_gradient += \
                reshaped_output_gradient[j, :, :].dot(
                    reshaped_kernel[i, j, :, :].T
                )
    
    # Gather the gradient of input
    for i in range(in_channels):
        for j in range(out_height):
            for k in range(out_width):
                row = j * stride
                column = k * stride
                input_gradient[i, row: row + stride, column: column + stride] += \
                    reshaped_input_gradient[i, j * out_width + k, :].reshape((ksize, ksize))
    
    input_gradient = input_gradient[:, pad: in_height + pad, pad: in_width + pad]
    
    return kernel_gradient, input_gradient


def maxpool(input, stride=2):
    
    in_channels, in_height, in_width = input.shape
    out_height, out_width = int(in_height / stride), int(in_width / stride)

    mat = np.zeros((in_channels, out_height * out_width, stride * stride))
    mask = np.zeros(input.shape)
    reshaped_mask = np.zeros(mat.shape)

    # construct mask
    for i in range(in_channels):
        for j in range(out_height):
            for k in range(out_width):
                row = j * stride
                column = k * stride
                mat[i, j * out_width + k, :] = \
                    input[i, row: row + stride, column: column + stride].reshape((1, -1))
                reshaped_mask[i, j * out_width + k, np.argmax(mat[i, j, :])] = 1
                mask[i, row: row + stride, column: column + stride] = \
                    reshaped_mask[i, j, :].reshape((stride, stride))
    
    output = np.sum(mat * reshaped_mask, axis=2).reshape(in_channels, out_height, out_width)

    return mask, output


def maxpool_deriviation(gradient, mask, stride):

    in_channels, out_height, out_width = gradient.shape
    dilated_gradient = np.zeros((in_channels, \
        int(out_height * stride), int(out_width * stride)))

    for i in range(in_channels):
        for j in range(out_height):
            for k in range(out_width):
                row = stride * j
                column = stride * k
                dilated_gradient[i, row: row + stride, column: column + stride] \
                    = gradient[i, j, k]

    return dilated_gradient * mask


def conv_initializer(kernel_shape):
    return np.random.random(kernel_shape) / np.prod(kernel_shape)


def naive_initializer(weights_shape):
    return np.random.random(weights_shape) / np.prod(weights_shape) + 1e-3
