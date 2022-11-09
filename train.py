#! /usr/bin/python
# -*- coding: utf8 -*-
import os
import tensorflow as tf

from deepsleeplite.trainer import DeepSleepNetLiteTrainer
from deepsleeplite.sleep_stages import (NUM_CLASSES,
                                        EPOCH_SEC_LEN,
                                        SEQ_OF_EPOCHS,
                                        SAMPLING_RATE)


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', 'data',
                           """Directory where to load training data.""")
tf.app.flags.DEFINE_string('output_dir', 'output',
                           """Directory where to save trained models """
                           """and outputs.""")
tf.app.flags.DEFINE_integer('n_folds', 1,
                           """Number of cross-validation folds.""")
tf.app.flags.DEFINE_integer('fold_idx', 0,
                            """Index of cross-validation fold to train.""")
tf.app.flags.DEFINE_integer('train_epochs', 100,
                            """Number of epochs for pretraining DeepFeatureNet.""")
tf.app.flags.DEFINE_float('smooth_value', 0.1,
                            """Alpha value for label smoothing.""")
tf.app.flags.DEFINE_boolean('smooth_stats', False,
                            """Whether to train with or without label smoothing with stats:
                            conditional probability distribution.""")
tf.app.flags.DEFINE_boolean('resume', False,
                            """Whether to resume the training process.""")



def train(n_epochs):
    trainer = DeepSleepNetLiteTrainer(
        data_dir=FLAGS.data_dir,
        output_dir=FLAGS.output_dir,
        n_folds=FLAGS.n_folds,
        fold_idx=FLAGS.fold_idx,
        batch_size=100,
        input_dims=EPOCH_SEC_LEN * SAMPLING_RATE,
        seq_length=SEQ_OF_EPOCHS,
        n_classes=NUM_CLASSES,
        interval_print_cm=5
    )
    trained_model_path = trainer.train(
        n_epochs=n_epochs,
        resume=FLAGS.resume,
        smooth_value=FLAGS.smooth_value,
        smooth_stats=FLAGS.smooth_stats
    )
    return trained_model_path

def main(argv=None):

    # Output dir
    output_dir = os.path.join(FLAGS.output_dir, "fold{}".format(FLAGS.fold_idx))

    if not FLAGS.resume:
        if tf.io.gfile.exists(output_dir):
            tf.io.gfile.rmtree(output_dir)
        tf.io.gfile.makedirs(output_dir)

    # FeatureNet
    trained_model_path = train(
    n_epochs=FLAGS.train_epochs
    )


if __name__ == "__main__":
    tf.compat.v1.app.run()
