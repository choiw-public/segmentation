from functions.utils import list_getter
from functions import loss_functions
from cv2 import VideoCapture, VideoWriter, VideoWriter_fourcc, imwrite
from math import pi, isnan, isinf
import importlib as imp
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import os
import time


class TrainHandler:
    """
    a parent class of ModelHandler
    """

    def _train_handler(self):
        if self.config.device == 'tpu':
            try:
                tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
                print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
            except ValueError:
                raise BaseException('ERROR: Not connected to a TPU runtime;')
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
            with tpu_strategy.scope():
                model = self.config.model_fn(self.config)
        else:
            model = self.config.model_fn(self.config)

        lr = tf.keras.experimental.LinearCosineDecay(initial_learning_rate=self.config.init_lr,
                                                     decay_steps=self.config.decay_steps,
                                                     num_periods=self.config.num_periods,
                                                     alpha=self.config.alpha,
                                                     beta=self.config.beta)
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)

        h, w = self.config.crop_size
        model.build(input_shape=(None, h, w, 3))
        epoch, step = 1, 1
        summary_writer = tf.summary.create_file_writer(os.path.join(self.config.model_dir, 'summary'))
        while step < self.config.max_step:
            avg_train_loss = 0
            for i, (X, Y) in enumerate(iter(self.train_data)):
                with tf.GradientTape() as g:
                    pred = model(X)
                    loss = loss_functions.miou_loss(Y, pred) + tf.reduce_sum(model.losses)  # model loss + l2 loss
                trainable_variables = model.trainable_variables
                gradients = g.gradient(loss, trainable_variables)
                optimizer.apply_gradients(zip(gradients, trainable_variables))
                avg_train_loss += loss
                if step % self.config.logging_step == 0:
                    print('Training epoch:%d, Step: %d, loss:%.3f, lr: %.9f' % (epoch, step, loss, optimizer._decayed_lr(tf.float32)), end='\r', flush=True)

                if step % self.config.val_step == 0:
                    avg_val_loss = 0
                    for j, (X_val, Y_val) in enumerate(iter(self.val_data)):
                        pred = model(X_val)
                        avg_val_loss += loss_functions.miou_loss(Y_val, pred)
                        print('Val loss @ epoch-%d: %.3f' % (epoch, loss), end='\r', flush=True)
                    with summary_writer.as_default():
                        tf.summary.scalar('val_loss', avg_val_loss / (j + 1), step=step)
                    print('Average val loss @ epoch-%d: %.3f' % (epoch, avg_val_loss / (j + 1)))

                if step % self.config.saving_step == 0:
                    model.save_weights(filepath=os.path.join(self.config.model_dir, 'model_ckpt', 'step-%06d' % step), save_format='tf')

                    print('model @ step-%d is saved' % step)

                if step % self.config.summary_step == 0:
                    with summary_writer.as_default():
                        tf.summary.scalar('train_loss', avg_train_loss / (i + 1), step=step)
                        tf.summary.scalar('learning rate', optimizer._decayed_lr(tf.float32), step=step)

                if step >= self.config.max_step:
                    with summary_writer.as_default():
                        tf.summary.scalar('train_loss', avg_train_loss / (i + 1), step=step)
                    break

                step += 1
            print('Average train loss @ epoch-%d: %.3f' % (epoch, avg_train_loss / (i + 1)))
            with summary_writer.as_default():
                tf.summary.scalar('train_loss', avg_train_loss / (i + 1), step=step)
            epoch += 1

    # import matplotlib.pyplot as plt
    # rndidx = np.random.randint(len(pred))
    # plt.subplot(2, 3, 1)
    # plt.imshow((X[rndidx, ::].numpy() * 255).astype(int))
    # plt.subplot(2, 3, 2)
    # plt.imshow(Y[rndidx, :, :, 0].numpy().astype(int))
    # plt.subplot(2, 3, 3)
    # plt.imshow(pred[rndidx, :, :, 0].numpy())
    # rndidx = np.random.randint(len(pred))
    # plt.subplot(2, 3, 4)
    # plt.imshow((X[rndidx, ::].numpy() * 255).astype(int))
    # plt.subplot(2, 3, 5)
    # plt.imshow(Y[rndidx, :, :, 0].numpy().astype(int))
    # plt.subplot(2, 3, 6)
    # plt.imshow(pred[rndidx, :, :, 0].numpy())


class EvalHandler:
    """
    a parent class of ModelHandler
    """

    def _get_ckpt_in_range(self):
        all_ckpt_list = [_.split(".index")[0] for _ in list_getter(self.config.model_dir, 'index')]
        ckpt_pattern = './model/checkpoints/model_step-%d'
        if self.config.ckpt_start == 'beginning':
            start_idx = 0
        else:
            start_idx = all_ckpt_list.index(ckpt_pattern % self.config.ckpt_start)

        if self.config.ckpt_end == 'end':
            end_idx = None
        else:
            end_idx = all_ckpt_list.index(ckpt_pattern % self.config.ckpt_end) + 1
        return all_ckpt_list[start_idx:end_idx:self.config.ckpt_step]

    def _calculate_segmentation_metric(self):
        tp = np.diag(self.cumulative_cmatrix)
        fp = np.sum(self.cumulative_cmatrix, axis=0) - tp
        fn = np.sum(self.cumulative_cmatrix, axis=1) - tp
        precision = tp / (tp + fp)  # precision of each class. [batch, class]
        recall = tp / (tp + fn)  # recall of each class. [batch, class]
        f1 = 2 * precision * recall / (precision + recall)
        iou = tp / (tp + fp + fn)  # iou of each class. [batch, class]
        miou = iou.mean()  # miou
        if iou.shape[0] <= 2:
            self.metrics = [precision[1], recall[1], f1[1], miou]
        else:
            self.metrics = [miou]

    def _write_eval_log(self, ckpt_id):
        with open(os.path.join(self.config.eval_log_dir, 'metric_overall.csv'), 'a+') as writer:
            writer.write('%s, ' % ckpt_id)
            writer.write(', '.join([str(value) for value in self.metrics]) + '\n')

    def _eval(self, sess, ckpt_id):
        while True:
            try:
                self.cumulative_cmatrix += sess.run(self.confusion_matrix)
            except tf.errors.OutOfRangeError:
                self._calculate_segmentation_metric()
                self._write_eval_log(ckpt_id)
                break

    def _eval_handler(self, sess):
        restorer = tf.train.Saver()
        pred = tf.expand_dims(tf.argmax(self.logit, 3), 3)
        self.confusion_matrix = tf.confusion_matrix(tf.reshape(self.gt, [-1]),
                                                    tf.reshape(pred, [-1]),
                                                    self.config.num_classes,
                                                    dtype=tf.float32)
        for ckpt in self._get_ckpt_in_range():
            self._init_log()
            self.cumulative_cmatrix = np.zeros((self.config.num_classes, self.config.num_classes))
            ckpt_id = os.path.basename(ckpt)
            if ckpt_id in [row.split(',')[0] for row in self.log[1:]]:
                print('Log for the current ckpt (%s) already exsit. This ckpt is skipped' % ckpt_id)
            else:
                print('Current ckpt: %s' % ckpt)
                restorer.restore(sess, ckpt)
                sess.run(self.data_init)
                self._eval(sess, ckpt_id)


class VisHandler:
    """
    a parent class of ModelHandler
    """

    def _get_ckpt(self):
        all_ckpt_list = [_.split(".index")[0] for _ in list_getter(self.config.model_dir, 'index')]
        ckpt_pattern = './model/checkpoints/model_step-%d'
        return all_ckpt_list[all_ckpt_list.index(ckpt_pattern % self.config.ckpt_id)]

    def _superimpose(self, image, pred):
        mask = np.ones_like(pred) - pred
        color_label = np.stack([np.zeros_like(pred), np.zeros_like(pred), pred * 255], 2)
        if self.config.data_type == "image":
            return image[:, :, ::-1] * np.expand_dims(mask, 2) + color_label
        elif self.config.data_type == "video":
            return image * np.expand_dims(mask, 2) + color_label

    def _vis_with_image(self, sess):
        while True:
            try:
                pred, image, filename = sess.run([self.pred,
                                                  tf.squeeze(self.input_data),
                                                  tf.squeeze(self.filename)])
                basename = os.path.basename(filename.decode("utf-8"))
                dst_name = self.config.vis_result_dir + "/" + basename
                imwrite(dst_name, self._superimpose(image, pred))
            except tf.errors.OutOfRangeError:
                break

    def _vis_with_video(self, sess):
        vid_list = list_getter(self.config.img_dir, ("avi", "mp4"))
        for vid_name in vid_list:
            vid = VideoCapture(vid_name)
            fps = round(vid.get(5))
            should_continue, frame = vid.read()
            basename = os.path.basename(vid_name)[:-4]
            dst_name = self.config.vis_result_dir + "/" + basename + ".avi"
            h, w, _ = frame.shape
            pred = sess.run(self.pred, {self.input_data: np.expand_dims(frame, 0)})
            superimposed = self._superimpose(frame, pred)
            vid_out = VideoWriter(dst_name, VideoWriter_fourcc(*"XVID"), fps, (w, h))
            vid_out.write(superimposed.astype(np.uint8))
            while should_continue:
                should_continue, frame = vid.read()
                if should_continue:
                    pred = sess.run(self.pred, {self.input_data: np.expand_dims(frame, 0)})
                    superimposed = self._superimpose(frame, pred)
                    vid_out.write(superimposed.astype(np.uint8))
            vid_out.release()

    def _vis_handler(self, sess):
        restorer = tf.train.Saver()
        self.pred = tf.squeeze(tf.argmax(self.logit, 3))
        restorer.restore(sess, self._get_ckpt())
        if self.config.data_type == "image":
            sess.run(self.data_init)
            self._vis_with_image(sess)
        elif self.config.data_type == "video":
            self._vis_with_video(sess)
        else:
            raise ValueError("Unexpected data_type")


class ModelHandler(TrainHandler, EvalHandler, VisHandler):
    def __init__(self, data, config):
        self.config = config
        self.train_data = data[0]
        self.val_data = data[1]
        super(ModelHandler, self).__init__()
        self._execute()

    @staticmethod
    def fp32_var_getter(getter,
                        name,
                        shape=None,
                        dtype=None,
                        initializer=None,
                        regularizer=None,
                        trainable=True,
                        *args, **kwargs):
        """Custom variable getter that forces trainable variables to be stored in
        float32 precision and then casts them to the training precision.
        """
        variable = getter(name,
                          shape,
                          dtype=tf.float32 if trainable else dtype,
                          initializer=initializer,
                          regularizer=regularizer,
                          trainable=trainable,
                          *args, **kwargs)
        if trainable and dtype != tf.float32:
            variable = tf.cast(variable, dtype)
        return variable

    def _execute(self):
        # Using the Winograd non-fused algorithms provides a small performance boost.
        if self.config.phase == "train":
            self._train_handler()
        elif self.config.phase == "eval":
            self._eval_handler(sess)
        elif self.config.phase == "vis":
            self._vis_handler(sess)
        else:
            raise ValueError("Unexpected phase:%s" % self.config.phase)
