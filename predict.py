#! /usr/bin/python
# -*- coding: utf8 -*-
import ntpath
import os
import time

from tensorflow.compat.v1 import ConfigProto

from datetime import datetime
import scipy
from sklearn.metrics import confusion_matrix, f1_score

from deepsleeplite.data_loader import DataLoader
from deepsleeplite.model import DeepSleepNetLite
from deepsleeplite.nn import *
from deepsleeplite.utils import *
from deepsleeplite.sleep_stages import (NUM_CLASSES,
                                        EPOCH_SEC_LEN,
                                        SAMPLING_RATE,
                                        DB_VERSION)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', 'data',
                           """Directory where to load training data.""")
tf.app.flags.DEFINE_string('model_dir', 'output',
                           """Directory where to load trained models.""")
tf.app.flags.DEFINE_string('output_dir', 'output',
                           """Directory where to save outputs.""")
tf.app.flags.DEFINE_boolean('cross_validation', True,
                            """Whether to predict for each cross-validation fold.""")
tf.app.flags.DEFINE_integer('n_folds', 20,
                           """ If crossvalidation True, define the number of folds.""")
tf.app.flags.DEFINE_boolean('MC_dropout', False,
                            """Whether to predict using Monte Carlo dropout.""")
tf.app.flags.DEFINE_integer('MC_sampling', 1,
                           """ Define the number of Monte Carlo dropout sampling. Set MC_sampling equal to 1
                            if you do not want to use the MC_dropout""")
tf.app.flags.DEFINE_boolean('smooth_stats', True,
                            """""")

coding2stages = {
    0 : "W",
    1 : "N1",
    2 : "N2",
    3 : "N3",
    4 : "R"
}

codingChange = {
    0 : 4,
    1 : 2,
    2 : 1,
    3 : 0,
    4 : 3
}


def print_performance(sess, network_name, n_examples, duration, loss, cm, acc, f1):
    # Get regularization loss
    reg_loss = tf.add_n(tf.get_collection("losses", scope=network_name + "\/"))
    reg_loss_value = sess.run(reg_loss)

    # Print performance
    print((
        "duration={:.3f} sec, n={}, loss={:.3f} ({:.3f}), acc={:.3f}, "
        "f1={:.3f}".format(
            duration, n_examples, loss, reg_loss_value, acc, f1
        )
    ))
    print(cm)
    print(" ")

def _run_epoch(
        sess,
        network,
        inputs,
        targets,
        seq_length,
        MC_sampling,
        train_op,
        output_dir,
        fold_idx

):
    start_time = time.time()
    y = []
    y_true = []
    y_selected = []
    y_true_selected = []

    y_var = []
    prob_pred = []
    prob_pred_selected = []

    query_instances = []
    correct_among_query = []

    total_loss, n_batches = 0.0, 0

    for sub_f_idx, each_data in enumerate(zip(inputs, targets)):

        each_x, each_y = each_data

        each_y_pred = []
        each_y_true = []
        each_prob_pred = []
        each_y_var = []

        start_time_each_rec = time.time()

        for x_batch, y_batch, y_batch_seq, batch_len, epochs_shifts in iterate_minibatches_prediction(
                                                                        inputs=each_x,
                                                                        targets=each_y,
                                                                        batch_size=network.batch_size,
                                                                        seq_length=seq_length):

            start_time_each_epoch = time.time()

            # Empty conditional probability distribution vector each batch
            y_distribution_batch = np.empty([network.batch_size, network.n_classes])

            # n times prediction
            prob_tmp = np.empty((MC_sampling, NUM_CLASSES))
            loss_value_tmp = np.empty((MC_sampling))
            for n in range(0, MC_sampling):
                feed_dict = {
                    network.input_var: x_batch,
                    network.target_var: y_batch,
                    network.alfa: 0.1,
                    network.conditional_distribution: y_distribution_batch
                }

                _, loss_value_tmp[n], y_pred, logits = sess.run(
                    [train_op, network.loss_op, network.pred_op, network.logits],
                    feed_dict=feed_dict)

                # Compute softmax probabilities values from logits
                prob_tmp[n, :] = scipy.special.softmax(logits[0])

            # Compute mean and variance
            mean_probs = np.mean(prob_tmp, axis=0)
            var_probs = np.var(prob_tmp, axis=0)
            # Compute the final prediction: max over the mean values
            y_pred = np.asarray([np.argmax(mean_probs)])
            # Compute mean loss
            mean_loss = np.mean(loss_value_tmp)

            each_y_pred.extend(y_pred)
            each_y_true.extend(y_batch)
            each_prob_pred.append(mean_probs)
            each_y_var.append(var_probs[y_pred[-1]])

            total_loss += mean_loss
            n_batches += 1

            # Check the loss value
            assert not np.isnan(mean_loss), \
                "Model diverged with loss = NaN"

        y.append(each_y_pred)
        y_true.append(each_y_true)
        prob_pred.append(each_prob_pred)
        y_var.append(each_y_var)

        duration_each_rec = time.time() - start_time_each_rec

        if MC_sampling != 1:

            n_examples = len(y_true[sub_f_idx])
            y_arr = np.asarray(y[sub_f_idx])
            y_true_arr = np.asarray(y_true[sub_f_idx])
            prob_pred_arr = np.asarray(prob_pred[sub_f_idx])
            y_var_arr = np.asarray(y_var[sub_f_idx])

            # Variance Rule selection - threshold 5% whole recording
            idx_threshold = int(0.95 * n_examples)
            var_threshold = np.sort(y_var_arr)[idx_threshold]
            removed_idx = np.where(y_var_arr >= var_threshold)[-1]
            selected_idx = np.where(y_var_arr < var_threshold)[-1]
            n_query = n_examples - len(selected_idx)

            query_instances.append(n_query)
            print('number of query {}'.format(n_query))
            correct_ = np.sum(y_true_arr[removed_idx] == y_arr[removed_idx])
            correct_among_query.append(correct_)
            print('number of which were correct {}'.format(correct_))

            y_selected.append(y_arr[selected_idx].tolist())
            y_true_selected.append(y_true_arr[selected_idx].tolist())
            prob_pred_selected.append(prob_pred_arr[selected_idx].tolist())

        else:

            query_instances.append(None)
            correct_among_query.append(None)
            y_selected.append(None)
            y_true_selected.append(None)
            prob_pred_selected.append(None)

    # # Save prediction

    save_dict = {
        "y_true": y_true,
        "y_pred": y,
        "prob_pred": prob_pred,
        "y_var": y_var,
        "y_true_selected": y_true_selected,
        "y_pred_selected": y_selected,
        "prob_pred_selected": prob_pred_selected,
        "query_instances": query_instances,
        "correct_among_query": correct_among_query
    }
    save_path = os.path.join(
        output_dir,
        "output_fold{}.npz".format(fold_idx)
    )

    np.savez(save_path, **save_dict)
    print("Saved outputs to {}".format(save_path))


    duration = time.time() - start_time
    total_loss /= n_batches
    total_y_pred = np.hstack(y)
    total_y_true = np.hstack(y_true)
    total_y_var = np.hstack(y_var)

    return total_y_true, total_y_pred, total_y_var, total_loss, duration


def predict_(
        data_dir,
        model_dir,
        output_dir,
        cross_validation,
        n_folds,
        MC_dropout,
        MC_sampling,
        smooth_stats
):

    # The model will be built into the default Graph
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=config) as sess:
        valid_net = DeepSleepNetLite(
            batch_size=1,
            input_dims=EPOCH_SEC_LEN * SAMPLING_RATE,
            seq_length=3,
            n_classes=NUM_CLASSES,
            is_train=False,
            reuse_params=False,
            MC_dropout=MC_dropout,
            smooth_stats=smooth_stats
        )

        # Initialize parameters
        valid_net.init_ops()

        if cross_validation:

            # Ground truth and predictions
            y_true = []
            y_pred = []
            accuracy = []

            for fold_idx in range(n_folds):

                checkpoint_path = os.path.join(
                    model_dir,
                    "fold{}".format(fold_idx),
                    "deepfeaturenet/checkpoint/"
                )

                if not os.path.exists(checkpoint_path):
                    accuracy.append('NaN')
                    continue

                # Restore the trained model
                saver = tf.train.Saver()
                saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
                print("Model restored from: {}\n".format(tf.train.latest_checkpoint(checkpoint_path)))

                # Load testing data -
                x, y, subjects_files, sampling_rate = DataLoader.load_data_cv_baseline(
                    data_dir=data_dir, fold_idx=fold_idx, version=DB_VERSION)

                # Loop each epoch
                print("[{}] Predicting ...\n".format(datetime.now()))

                # # Evaluate the model on the subject data
                y_true_, y_pred_, y_var_, loss, duration = \
                    _run_epoch(
                        sess=sess, network=valid_net,
                        inputs=x, targets=y,
                        seq_length=3,
                        MC_sampling=MC_sampling,
                        train_op=tf.no_op(),
                        output_dir=output_dir,
                        fold_idx=fold_idx
                    )

                n_examples = len(y_true)

                cm_ = confusion_matrix(y_true_, y_pred_)
                acc_ = np.mean(y_true_ == y_pred_)
                mf1_ = f1_score(y_true_, y_pred_, average="weighted")

                # Report performance
                print_performance(
                    sess, valid_net.name,
                    n_examples, duration, loss,
                    cm_, acc_, mf1_
                )

                accuracy.append(acc_)

                y_true.extend(y_true_)
                y_pred.extend(y_pred_)

            # Overall performance
            print("[{}] Overall prediction performance\n".format(datetime.now()))
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)
            n_examples = len(y_true)
            cm = confusion_matrix(y_true, y_pred)
            acc = np.mean(y_true == y_pred)
            mf1 = f1_score(y_true, y_pred, average="weighted")
            print((
                "n={}, acc={:.3f}, f1={:.3f}".format(
                    n_examples, acc, mf1
                )
            ))
            print(cm)

        else:

            fold_idx = 0

            checkpoint_path = os.path.join(
                model_dir,
                "fold{}".format(fold_idx),
                "deepsleepnetlite/checkpoint/"
            )

            assert os.path.exists(checkpoint_path), \
                "The Model does not exist"

            # Restore the trained model
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
            print("Model restored from: {}\n".format(tf.train.latest_checkpoint(checkpoint_path)))

            # Load test files from existing data files -
            path_data_files = os.path.join(model_dir, "fold{}".format(fold_idx), "deepsleepnetlite/data_file{}.npz".format(fold_idx))
            with np.load(path_data_files, allow_pickle=True) as f:
                test_files = f["test_files"]

            # Loop each epoch
            print("[{}] Predicting ...\n".format(datetime.now()))

            # # Evaluate the model on the subject data
            y_true, y_pred, y_var, loss, duration = \
                _run_epoch(
                    sess=sess, network=valid_net,
                    inputs=test_files,
                    seq_length=3,
                    MC_sampling=MC_sampling,
                    train_op=tf.no_op(),
                    output_dir=output_dir
                )


            # Overall performance
            print("[{}] Overall prediction performance\n".format(datetime.now()))
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)
            n_examples = len(y_true)
            cm = confusion_matrix(y_true, y_pred)
            acc = np.mean(y_true == y_pred)
            mf1 = f1_score(y_true, y_pred, average="weighted")
            print((
                "duration={:.3f}, n={}, acc={:.3f}, f1={:.3f}".format(
                    duration, n_examples, acc, mf1
                )
            ))
            print(cm)





def main(argv=None):

    # Output dir
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    predict_(
        data_dir=FLAGS.data_dir,
        model_dir=FLAGS.model_dir,
        output_dir=FLAGS.output_dir,
        cross_validation=FLAGS.cross_validation,
        n_folds=FLAGS.n_folds,
        MC_dropout=FLAGS.MC_dropout,
        MC_sampling=FLAGS.MC_sampling,
        smooth_stats=FLAGS.smooth_stats
    )


if __name__ == "__main__":
    tf.app.run()