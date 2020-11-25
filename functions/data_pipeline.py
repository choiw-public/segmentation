from functions.preprocess import Preprocessing
from functions.utils import list_getter
import tensorflow as tf
import os

auto = tf.data.experimental.AUTOTUNE


class DataPipeline(Preprocessing):
    def __init__(self, tfrecord_dir, batch_size, aug_config, is_train_set):
        self.tfrecord_feature = {"image": tf.io.FixedLenFeature((), tf.string),
                                 "height": tf.io.FixedLenFeature((), tf.int64),
                                 "width": tf.io.FixedLenFeature((), tf.int64),
                                 "segmentation": tf.io.FixedLenFeature((), tf.string)}
        self.tfrecord_dir = tfrecord_dir
        self.batch_size = batch_size
        self.aug_config = aug_config
        self.is_train_set = is_train_set

    def build(self):
        tfrecord_list = list_getter(self.tfrecord_dir, extension="tfrecord")
        if not tfrecord_list:
            raise ValueError("tfrecord does not exist: %s" % self.tfrecord_dir)
        data = tf.data.TFRecordDataset(tfrecord_list, num_parallel_reads=auto)
        data = data.map(self._tfrecord_parser, auto)
        if self.is_train_set:
            data = data.shuffle(self.batch_size * 10)
        data = data.prefetch(auto)
        data = data.batch(self.batch_size, drop_remainder=self.is_train_set)
        return data

    def _tfrecord_parser(self, data):
        parsed = tf.io.parse_single_example(data, self.tfrecord_feature)
        image = tf.convert_to_tensor(tf.image.decode_jpeg(parsed["image"], channels=3))
        gt = tf.convert_to_tensor(tf.image.decode_jpeg(parsed["segmentation"], channels=1))
        image, gt = self.preprocessing(image, gt)
        return image, gt

    @staticmethod
    def _image_gt_parser(image_name, gt_name):
        image = tf.image.decode_png(tf.read_file(image_name), 3)
        gt = tf.image.decode_png(tf.read_file(gt_name), 1)
        return {"input_data": image, "gt": gt, "filename": image_name}

    @staticmethod
    def _image_parser(image_name):
        return {"input_data": tf.image.decode_png(tf.read_file(image_name), 3), "filename": image_name}

    def _input_from_tfrecord(self):
        batch_size = self.config.batch_size
        # for main data pipeline
        return self._get_batch_and_init(self.config.train_tfrecord_dir, batch_size)

    def _input_from_image(self):
        def inspect_file_extension(target_list):
            extensions = list(set([os.path.basename(img_name).split(".")[-1] for img_name in target_list]))
            if len(extensions) > 1:
                raise ValueError("Multiple image formats are used:")
            elif len(extensions) == 0:
                raise ValueError("no image files exist")

        def inspect_pairness(list1, list2):
            if not len(list1) == len(list2):
                raise ValueError("number of images are different")
            for file1, file2 in zip(list1, list2):
                file1_name = os.path.basename(file1).split(".")[-2]
                file2_name = os.path.basename(file2).split(".")[-2]
                if not file1_name == file2_name:
                    raise ValueError("image names are different: %s | %s" % (file2, file1))

        img_list = list_getter(self.config.img_dir, "jpg")
        img_list_tensor = tf.convert_to_tensor(img_list, dtype=tf.string)
        img_data = tf.data.Dataset.from_tensor_slices(img_list_tensor)
        if self.config.phase == "eval":
            gt_list = list_getter(self.seg_dir, "png")
            inspect_pairness(gt_list, img_list)
            inspect_file_extension(gt_list)
            inspect_file_extension(img_list)
            gt_list_tensor = tf.convert_to_tensor(gt_list, dtype=tf.string)
            gt_data = tf.data.Dataset.from_tensor_slices(gt_list_tensor)
            data = tf.data.Dataset.zip((img_data, gt_data))
            data = data.map(self._image_gt_parser, 4).batch(self.config.batch_size, False)
        else:
            data = img_data.map(self._image_parser, 4).batch(self.config.batch_size, False)
        data = data.prefetch(4)  # tf.data_pipeline.experimental.AUTOTUNE
        iterator = data.make_initializable_iterator()
        dataset = iterator.get_next()
        self.input_data = dataset["input_data"]
        self.gt = dataset["gt"] if self.config.phase == "eval" else None
        self.filename = dataset["filename"]
        self.data_init = iterator.initializer


def get_datasets(config):
    train_data = DataPipeline(tfrecord_dir=config.train_tfrecord_dir,
                              batch_size=config.train_batch_size,
                              aug_config=config,
                              is_train_set=True).build()
    if config.val_tfrecord_dir:
        val_data = DataPipeline(tfrecord_dir=config.val_tfrecord_dir,
                                batch_size=config.val_batch_size,
                                aug_config=None,
                                is_train_set=True).build()
    else:
        val_data = None
    return train_data, val_data
