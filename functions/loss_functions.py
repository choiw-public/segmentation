from functions.utils import get_shape
import tensorflow as tf


def miou_loss(gt, logit):
    # calculated bache mean intersection over union loss
    prob_map = tf.nn.softmax(logit)
    onehot_gt = tf.one_hot(tf.cast(tf.squeeze(gt, 3), tf.uint8), get_shape(logit)[-1])
    if prob_map.shape.as_list() != onehot_gt.shape.as_list():
        raise ValueError('dimension mismatching')
    # calculate iou loss
    intersection_logit = prob_map * onehot_gt  # [batch, height, width, class]
    union_logit = prob_map + onehot_gt - intersection_logit  # [batch, height, width, class]
    iou_logit = tf.reduce_sum(intersection_logit, [0, 1, 2]) / tf.reduce_sum(union_logit, [0, 1, 2])  # class
    miou_logit = tf.reduce_mean(iou_logit)
    return 1.0 - tf.reduce_mean(miou_logit)


def dice_loss(gt, logit):
    gt = tf.cast(gt, tf.float32)
    pred = tf.math.sigmoid(logit)
    numerator = 2 * tf.reduce_sum(gt * pred)
    denominator = tf.reduce_sum(gt + pred)
    return 1 - numerator / denominator
