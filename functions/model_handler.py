from functions.utils import list_getter
from functions import loss_functions
from cv2 import VideoCapture, VideoWriter, VideoWriter_fourcc, imwrite
from functions.data_pipeline import get_datasets
from math import pi, isnan, isinf
import importlib as imp

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import numpy as np
import time
import os


class TrainHandler:
    def __init__(self, model, config):
        self.train_data, self.val_data = get_datasets(config)
        lr = tf.keras.experimental.LinearCosineDecay(initial_learning_rate=config.init_lr,
                                                     decay_steps=config.decay_steps,
                                                     num_periods=config.num_periods,
                                                     alpha=config.alpha,
                                                     beta=config.beta)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
        self.summary_writer = tf.summary.create_file_writer(os.path.join(config.model_dir, 'summary'))
        self.model_name = config.model_name
        self.model = model
        self.max_step = config.max_step
        self.logging_step = config.logging_step
        self.val_step = config.val_step
        self.saving_step = config.saving_step
        self.summary_step = config.summary_step
        self.model_dir = config.model_dir
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(0), epoch=tf.Variable(0), optimizer=self.optimizer,
                                        net=self.model)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt,
                                                       os.path.join(self.model_dir, 'saved_ckpts'),
                                                       max_to_keep=None)
        self.mixup_prob = config.mixup_prob
        self.mixup_holder = []
        self.mixup_dist_fn = tfp.distributions.Beta(0.2, 0.2)
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print('Last checkpoint is restored and training is continued')

    def _feed(self, x, y, training):
        pred = self.model(x, training)
        loss = loss_functions.miou_loss(y, pred) + tf.reduce_sum(self.model.losses)  # model loss + l2 loss
        return pred, loss

    def _train_step(self, train_x, train_y):
        with tf.GradientTape() as g:
            _, loss = self._feed(train_x, train_y, True)
        trainable_variables = self.model.trainable_variables
        gradients = g.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        self.ckpt.step.assign_add(1)
        return loss

    def _val_step(self, val_x, val_y):
        _, loss = self._feed(val_x, val_y, False)
        return loss

    def _write_scalar_summary(self, name, value):
        with self.summary_writer.as_default():
            tf.summary.scalar(name, value, step=self.ckpt.step.numpy())

    def _mixup(self, x, y):
        if np.random.uniform(0, 1) >= self.mixup_prob:
            if self.mixup_holder:
                x_tmp = self.mixup_holder[0][0]
                y_tmp = self.mixup_holder[0][1]
                dist = self.mixup_dist_fn.sample([len(x_tmp), 1, 1, 1])
                x_new = dist * x + (1 - dist) * x_tmp
                y_new = dist * y + (1 - dist) * y_tmp
                self.mixup_holder.pop()
                self.mixup_holder.append([x, y])
                return x_new, y_new
            else:
                self.mixup_holder.append([x, y])
                return x, y
        return x, y

    def train(self):
        while self.ckpt.step < self.max_step:
            avg_train_loss = 0
            tic = time.time()
            for i, (x, y) in enumerate(iter(self.train_data)):
                x, y = self._mixup(x, y)
                loss = self._train_step(x, y)
                avg_train_loss += loss
                if self.ckpt.step % self.logging_step == 0:
                    print('Model name: %s, Epoch:%d, Step: %d, Loss:%.3f, lr: %.9f'
                          % (self.model_name, self.ckpt.epoch, self.ckpt.step, loss,
                             self.optimizer._decayed_lr(tf.float32)), end='\r', flush=True)

                if self.ckpt.step % self.val_step == 0:
                    avg_val_loss = 0
                    for j, (x_val, y_val) in enumerate(iter(self.val_data)):
                        avg_val_loss += self._val_step(x_val, y_val)
                        print('Val loss @ epoch-%d: %.3f' % (self.ckpt.epoch, loss), end='\r')
                    avg_val_loss /= (j + 1)
                    self._write_scalar_summary('val_loss', avg_val_loss)
                    print('Average val loss @ epoch-%d: %.3f \n' % (self.ckpt.epoch, avg_val_loss / (j + 1)))

                if self.ckpt.step % self.saving_step == 0:
                    self.ckpt_manager.save(checkpoint_number=self.ckpt.step)
                    print('%s @ step-%d is saved' % (self.model_name, self.ckpt.step))

                if self.ckpt.step % self.summary_step == 0:
                    self._write_scalar_summary('train_loss', avg_train_loss / (i + 1))
                    self._write_scalar_summary('learning_rate', self.optimizer._decayed_lr(tf.float32))

                if self.ckpt.step >= self.max_step:
                    self._write_scalar_summary('train_loss', avg_train_loss / (i + 1))
                    break
            print('Average train loss @ epoch-%d (step-%d): %.3f | %.3f sec/epoch'
                  % (self.ckpt.epoch, self.ckpt.step - 1, avg_train_loss / (i + 1), time.time() - tic))
            self._write_scalar_summary('train_loss', avg_train_loss / (i + 1))
            self.ckpt.epoch.assign_add(1)


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


class ModelHandler(EvalHandler, VisHandler):
    def __init__(self, config):
        if config.device == 'tpu':
            try:
                tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
                print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
            except ValueError:
                raise BaseException('ERROR: Not connected to a TPU runtime;')
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
            with tpu_strategy.scope():
                self.model = config.model_fn(config)
        else:
            self.model = config.model_fn(config)
        if config.phase == 'train':
            TrainHandler(self.model, config).train()

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
