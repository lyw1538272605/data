"""
Typical use:
   from tensorflow.contrib.slim.nets import resnet_v2
ResNet-101 for image classification into 1000 classes:
   # inputs has shape [batch, 224, 224, 3]
   with slim.arg_scope(resnet_v2.resnet_arg_scope(is_training)):
      net, end_points = resnet_v2.resnet_v2_101(inputs, 1000)
ResNet-101 for semantic segmentation into 21 classes:
   # inputs has shape [batch, 513, 513, 3]
   with slim.arg_scope(resnet_v2.resnet_arg_scope(is_training)):
      net, end_points = resnet_v2.resnet_v2_101(inputs,
                                                21,
                                                global_pool=False,
                                                output_stride=16)
"""
import collections
import tensorflow as tf
import numpy as np
import pandas as pd
import dataread as dr

slim = tf.contrib.slim

class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    """A named tuple describing a ResNet block.
    Its parts are:
      scope: The scope of the `Block`.
      unit_fn: The ResNet unit function which takes as input a `Tensor` and
        returns another `Tensor` with the output of the ResNet unit.
      args: A list of length equal to the number of units in the `Block`. The list
        contains one (depth, depth_bottleneck, stride) tuple for each unit in the
        block to serve as argument to unit_fn.
    """


def subsample(inputs, factor, scope=None):
    """Subsamples the input along the spatial dimensions.
    Args:
      inputs: A `Tensor` of size [batch, height_in, width_in, channels].
      factor: The subsampling factor.
      scope: Optional variable_scope.
    Returns:
      output: A `Tensor` of size [batch, height_out, width_out, channels] with the
        input, either intact (if factor == 1) or subsampled (if factor > 1).
    """
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)


def conv2d_same(inputs, num_outputs, kernel_size, stride, scope=None):
    """Strided 2-D convolution with 'SAME' padding.
    When stride > 1, then we do explicit zero-padding, followed by conv2d with
    'VALID' padding.
    Note that
       net = conv2d_same(inputs, num_outputs, 3, stride=stride)
    is equivalent to
       net = slim.conv2d(inputs, num_outputs, 3, stride=1, padding='SAME')
       net = subsample(net, factor=stride)
    whereas
       net = slim.conv2d(inputs, num_outputs, 3, stride=stride, padding='SAME')
    is different when the input's height or width is even, which is why we add the
    current function. For more details, see ResnetUtilsTest.testConv2DSameEven().
    Args:
      inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
      num_outputs: An integer, the number of output filters.
      kernel_size: An int with the kernel_size of the filters.
      stride: An integer, the output stride.
      rate: An integer, rate for atrous convolution.
      scope: Scope.
    Returns:
      output: A 4-D tensor of size [batch, height_out, width_out, channels] with
        the convolution output.
    """
    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1,
                           padding='SAME', scope=scope)
    else:
        # kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs,
                        [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                           padding='VALID', scope=scope)


@slim.add_arg_scope
def stack_blocks_dense(net, blocks,
                       outputs_collections=None):
    """Stacks ResNet `Blocks` and controls output feature density.
    First, this function creates scopes for the ResNet in the form of
    'block_name/unit_1', 'block_name/unit_2', etc.
    Args:
      net: A `Tensor` of size [batch, height, width, channels].
      blocks: A list of length equal to the number of ResNet `Blocks`. Each
        element is a ResNet `Block` object describing the units in the `Block`.
      outputs_collections: Collection to add the ResNet block outputs.
    Returns:
      net: Output tensor
    """
    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            for i, unit in enumerate(block.args):
                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                    unit_depth, unit_depth_bottleneck, unit_stride = unit
                    net = block.unit_fn(net,
                                        depth=unit_depth,
                                        depth_bottleneck=unit_depth_bottleneck,
                                        stride=unit_stride)
            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

    return net


def resnet_arg_scope(is_training=True,
                     weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
    """Defines the default ResNet arg scope.
    TODO(gpapan): The batch-normalization related default values above are
      appropriate for use in conjunction with the reference ResNet models
      released at https://github.com/KaimingHe/deep-residual-networks. When
      training ResNets from scratch, they might need to be tuned.
    Args:
      is_training: Whether or not we are training the parameters in the batch
        normalization layers of the model.
      weight_decay: The weight decay to use for regularizing the model.
      batch_norm_decay: The moving average decay when estimating layer activation
        statistics in batch normalization.
      batch_norm_epsilon: Small constant to prevent division by zero when
        normalizing activations by their variance in batch normalization.
      batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
        activations in the batch normalization layer.
    Returns:
      An `arg_scope` to use for the resnet models.
    """
    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            # The following implies padding='SAME' for pool1, which makes feature
            # alignment easier for dense prediction tasks. This is also used in
            # https://github.com/facebook/fb.resnet.torch. However the accompanying
            # code of 'Deep Residual Learning for Image Recognition' uses
            # padding='VALID' for pool1. You can switch to that choice by setting
            # slim.arg_scope([slim.max_pool2d], padding='VALID').
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc


@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride,
               outputs_collections=None, scope=None):
    """Bottleneck residual unit variant with BN before convolutions.
    This is the full preactivation residual unit variant proposed in [2]. See
    Fig. 1(b) of [2] for its definition. Note that we use here the bottleneck
    variant which has an extra bottleneck layer.
    When putting together two consecutive ResNet blocks that use this unit, one
    should use stride = 2 in the last unit of the first block.
    Args:
      inputs: A tensor of size [batch, height, width, channels].
      depth: The depth of the ResNet unit output.
      depth_bottleneck: The depth of the bottleneck layers.
      stride: The ResNet unit's stride. Determines the amount of downsampling of
        the units output compared to its input.
      rate: An integer, rate for atrous convolution.
      outputs_collections: Collection to add the ResNet unit output.
      scope: Optional variable_scope.
    Returns:
      The ResNet unit's output.
    """
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                                   normalizer_fn=None, activation_fn=None,
                                   scope='shortcut')

        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1,
                               scope='conv1')
        residual = conv2d_same(residual, depth_bottleneck, 3, stride,
                               scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                               normalizer_fn=None, activation_fn=None,
                               scope='conv3')

        output = shortcut + residual

        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.name,
                                                output)


def resnet_v2(inputs,
              blocks,
              num_classes=None,
              global_pool=True,
              include_root_block=True,
              reuse=None,
              scope=None):
    """Generator for v2 (preactivation) ResNet models.
    This function generates a family of ResNet v2 models. See the resnet_v2_*()
    methods for specific model instantiations, obtained by selecting different
    block instantiations that produce ResNets of various depths.
    Args:
      inputs: A tensor of size [batch, height_in, width_in, channels].
      blocks: A list of length equal to the number of ResNet blocks. Each element
        is a resnet_utils.Block object describing the units in the block.
      num_classes: Number of predicted classes for classification tasks. If None
        we return the features before the logit layer.
      include_root_block: If True, include the initial convolution followed by
        max-pooling, if False excludes it. If excluded, `inputs` should be the
        results of an activation-less convolution.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.
    Returns:
      net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
        If global_pool is False, then height_out and width_out are reduced by a
        factor of output_stride compared to the respective height_in and width_in,
        else both height_out and width_out equal one. If num_classes is None, then
        net is the output of the last ResNet block, potentially after global
        average pooling. If num_classes is not None, net contains the pre-softmax
        activations.
      end_points: A dictionary from components of the network to the corresponding
        activation.
    Raises:
      ValueError: If the target output_stride is not valid.
    """
    with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck,
                             stack_blocks_dense],
                            outputs_collections=end_points_collection):
            net = inputs
            if include_root_block:
                # We do not include batch normalization or activation functions in conv1
                # because the first ResNet unit will perform these. Cf. Appendix of [2].
                with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
                    net = conv2d_same(net, 64, 7, stride=2, scope='conv1')

                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
            net = stack_blocks_dense(net, blocks)
            # This is needed because the pre-activation variant does not have batch
            # normalization or activation functions in the residual unit output. See
            # Appendix of [2].
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
            if global_pool:
                # Global average pooling.
                net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
            if num_classes is not None:
                net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                  normalizer_fn=None, scope='logits')
            # Convert end_points_collection into a dictionary of end_points.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if num_classes is not None:
                end_points['predictions'] = slim.softmax(net, scope='predictions')
            return net, end_points

def resnet_v2_my(inputs,
                 num_classes=None,
                 global_pool=True,
                 reuse=None,
                 scope='resnet_v2_50'):
    """ResNet-50 model of [1]. See resnet_v2() for arg and return description."""
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 1 + [(256, 64, 2)]),
        Block(
            'block2', bottleneck, [(512, 128, 1)] * 2 + [(512, 128, 2)]),
        Block(
            'block3', bottleneck, [(1024, 256, 1)] * 2 + [(1024, 256, 2)]),
        Block(
            'block4', bottleneck, [(2048, 512, 1)] * 1)]
    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=True, reuse=reuse, scope=scope)


def resnet_v2_50(inputs,
                 num_classes=None,
                 global_pool=True,
                 reuse=None,
                 scope='resnet_v2_50'):
    """ResNet-50 model of [1]. See resnet_v2() for arg and return description."""
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block(
            'block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block(
            'block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
        Block(
            'block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=True, reuse=reuse, scope=scope)


def resnet_v2_101(inputs,
                  num_classes=None,
                  global_pool=True,
                  reuse=None,
                  scope='resnet_v2_101'):
    """ResNet-101 model of [1]. See resnet_v2() for arg and return description."""
    blocks = [
        Block(
            'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block(
            'block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block(
            'block3', bottleneck, [(1024, 256, 1)] * 22 + [(1024, 256, 2)]),
        Block(
            'block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=True, reuse=reuse, scope=scope)


def resnet_v2_152(inputs,
                  num_classes=None,
                  global_pool=True,
                  reuse=None,
                  scope='resnet_v2_152'):
    """ResNet-152 model of [1]. See resnet_v2() for arg and return description."""
    blocks = [
        Block(
            'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block(
            'block2', bottleneck, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
        Block(
            'block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        Block(
            'block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=True, reuse=reuse, scope=scope)


def resnet_v2_200(inputs,
                  num_classes=None,
                  global_pool=True,
                  reuse=None,
                  scope='resnet_v2_200'):
    """ResNet-200 model of [2]. See resnet_v2() for arg and return description."""
    blocks = [
        Block(
            'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block(
            'block2', bottleneck, [(512, 128, 1)] * 23 + [(512, 128, 2)]),
        Block(
            'block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        Block(
            'block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=True, reuse=reuse, scope=scope)


# 生成一个训练batch
batch_i = 0
def get_next_batch(batch_size=64):
    global batch_i

    if (batch_i + 1) * batch_size > len(train):
        end = len(train)
        batch_i = -1
    else:
        end = (batch_i + 1) * batch_size
    batch_x = train[batch_i * batch_size : end]
    batch_y = train_labels[batch_i * batch_size: end]

    batch_i = batch_i + 1
    return batch_x, batch_y


####################################################################
#获取训练数据


# 训练
max_acc = 0.71 #保存model精度要求
train, validation, train_labels, validation_labels = dr.read()

def train_crack_captcha_cnn(is_train, checkpoint_dir):
    global max_acc
    X = tf.placeholder(tf.float32, [None, dr.ROWS, dr.COLS, dr.CHANNELS])
    Y = tf.placeholder(tf.float32, [None, 1, 1, 2])
    output, end_points = resnet_v2_50(X, num_classes = 2)

    global_steps = tf.Variable(1, trainable=False)
    learning_rate = tf.train.exponential_decay(0.0001, global_steps, 100, 0.9)

    with tf.device('/device:GPU:0'):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=output))
        # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_steps)

    predict = tf.argmax(output, axis = 3)
    l = tf.argmax(Y, axis = 3)
    correct_pred = tf.equal(predict, l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    ## tensorboard
    acc_summary = tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("learning_rate", learning_rate)

    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement = True)) as sess:
        if is_train:
            train_writer = tf.summary.FileWriter("/tmp/cnn_log/train", graph = sess.graph)
            test_writer = tf.summary.FileWriter("/tmp/cnn_log/test")

            sess.run(tf.global_variables_initializer())
            step_value = sess.run(global_steps)
            while step_value < 10000:
                step_value = sess.run(global_steps)
                merged = tf.summary.merge_all()
                batch_x, batch_y = get_next_batch()

                result, _, _loss= sess.run([merged, optimizer, loss], feed_dict={X: batch_x, Y: batch_y})
                train_writer.add_summary(result, step_value)
                print('step : {} loss : {}'.format(step_value, _loss))

                # 每100 step计算一次准确率
                if step_value % 10 == 0:
                    test_acc, acc = sess.run([acc_summary, accuracy], feed_dict={X: validation, Y: validation_labels})
                    test_writer.add_summary(test_acc, step_value)
                    print('accuracy : {}'.format(acc))

                    # 如果准确率大于max_acc,保存模型,完成训练
                    if acc > max_acc:
                        max_acc = float(acc) #转换类型防止变为同一个引用
                        saver.save(sess, checkpoint_dir + "/" + str(step_value) + '-' + str(acc) + "/model.ckpt", global_step=global_steps)

                        ##### predict #####
                        # predict_y = sess.run(output, feed_dict={X: test})
                        # data = pd.DataFrame([i for i in range(1, len(predict_y) + 1)], columns = ['id'])
                        # predict_y = np.argmax(predict_y, axis = 3)
                        # predict_y = np.reshape(predict_y,(-1))
                        # print(predict_y)
                        # predict_y = pd.Series(predict_y, name='label')
                        # data['label'] = predict_y
                        # data.to_csv("gender_submission.csv" + str(step), index=False)
                        ##### end #####
            train_writer.close()
        else:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                acc = sess.run(accuracy, feed_dict={X: validation, Y: validation_labels})
                print('accuracy : {}'.format(acc))

train_crack_captcha_cnn(True, 'check')
