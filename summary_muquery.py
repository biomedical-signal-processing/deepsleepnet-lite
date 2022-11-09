#! /usr/bin/python
# -*- coding: utf8 -*-

import argparse
import os
import re

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from tensorflow_probability.python.internal import dtype_util

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score

from deepsleeplite.sleep_stages import W, N1, N2, N3, REM

coding2stages = {
    0 : "W",
    1 : "N1",
    2 : "N2",
    3 : "N3",
    4 : "R"
}

coding2col = {
    0 : "k",
    1 : "green",
    2 : "r",
    3 : "b",
    4 : "c"
}

codingChange = {
    0 : 4,
    1 : 2,
    2 : 1,
    3 : 0,
    4 : 3
}

new_codingChange = {
    0 : 0,
    1 : 2,
    2 : 3,
    3 : 4,
    4 : 1
}

def _compute_calibration_bin_statistics(num_bins, logits=None, labels_true=None, labels_predicted=None):
  """Compute binning statistics required for calibration measures.
  Args:
    num_bins: int, number of probability bins, e.g. 10.
    logits: Tensor, (n,nlabels), with logits for n instances and nlabels.
    labels_true: Tensor, (n,), with tf.int32 or tf.int64 elements containing
      ground truth class labels in the range [0,nlabels].
    labels_predicted: Tensor, (n,), with tf.int32 or tf.int64 elements
      containing decisions of the predictive system.  If `None`, we will use
      the argmax decision using the `logits`.
  Returns:
    bz: Tensor, shape (2,num_bins), tf.int32, counts of incorrect (row 0) and
      correct (row 1) predictions in each of the `num_bins` probability bins.
    pmean_observed: Tensor, shape (num_bins,), tf.float32, the mean predictive
      probabilities in each probability bin.
  """

  if labels_predicted is None:
    # If no labels are provided, we take the label with the maximum probability
    # decision.  This corresponds to the optimal expected minimum loss decision
    # under 0/1 loss.
    pred_y = tf.argmax(logits, axis=1, output_type=labels_true.dtype)
  else:
    pred_y = labels_predicted

  correct = tf.cast(tf.equal(pred_y, labels_true), tf.int32)

  # Collect predicted probabilities of decisions
  # pred = tf.nn.softmax(logits, axis=1)
  pred = logits
  prob_y = tf1.batch_gather(pred, pred_y[:, tf.newaxis])  # p(pred_y | x)
  prob_y = tf.reshape(prob_y, (tf.size(prob_y),))

  # Compute b/z histogram statistics:
  # bz[0,bin] contains counts of incorrect predictions in the probability bin.
  # bz[1,bin] contains counts of correct predictions in the probability bin.
  bins = tf.histogram_fixed_width_bins(prob_y, [0.0, 1.0], nbins=num_bins)
  event_bin_counts = tf.math.bincount(
      correct * num_bins + bins,
      minlength=2 * num_bins,
      maxlength=2 * num_bins)
  event_bin_counts = tf.reshape(event_bin_counts, (2, num_bins))

  # Compute mean predicted probability value in each of the `num_bins` bins
  pmean_observed = tf.math.unsorted_segment_sum(prob_y, bins, num_bins)
  tiny = np.finfo(dtype_util.as_numpy_dtype(logits.dtype)).tiny
  pmean_observed = pmean_observed / (
      tf.cast(tf.reduce_sum(event_bin_counts, axis=0), logits.dtype) + tiny)

  return event_bin_counts, pmean_observed

def expected_calibration_error(num_bins, logits=None, labels_true=None, labels_predicted=None, name=None):
  """Compute the Expected Calibration Error (ECE).
  This method implements equation (3) in [1].  In this equation the probability
  of the decided label being correct is used to estimate the calibration
  property of the predictor.
  Note: a trade-off exist between using a small number of `num_bins` and the
  estimation reliability of the ECE.  In particular, this method may produce
  unreliable ECE estimates in case there are few samples available in some bins.
  As an alternative to this method, consider also using
  `bayesian_expected_calibration_error`.
  #### References
  [1]: Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger,
       On Calibration of Modern Neural Networks.
       Proceedings of the 34th International Conference on Machine Learning
       (ICML 2017).
       arXiv:1706.04599
       https://arxiv.org/pdf/1706.04599.pdf
  Args:
    num_bins: int, number of probability bins, e.g. 10.
    logits: Tensor, (n,nlabels), with logits for n instances and nlabels.
    labels_true: Tensor, (n,), with tf.int32 or tf.int64 elements containing
      ground truth class labels in the range [0,nlabels].
    labels_predicted: Tensor, (n,), with tf.int32 or tf.int64 elements
      containing decisions of the predictive system.  If `None`, we will use
      the argmax decision using the `logits`.
    name: Python `str` name prefixed to Ops created by this function.
  Returns:
    ece: Tensor, scalar, tf.float32.
  """
  with tf.name_scope(name or 'expected_calibration_error'):
    logits = tf.convert_to_tensor(logits)
    labels_true = tf.convert_to_tensor(labels_true)
    if labels_predicted is not None:
      labels_predicted = tf.convert_to_tensor(labels_predicted)

    # Compute empirical counts over the events defined by the sets
    # {incorrect,correct}x{0,1,..,num_bins-1}, as well as the empirical averages
    # of predicted probabilities in each probability bin.
    event_bin_counts, pmean_observed = _compute_calibration_bin_statistics(
        num_bins, logits=logits, labels_true=labels_true,
        labels_predicted=labels_predicted)

    # Compute the marginal probability of observing a probability bin.
    event_bin_counts = tf.cast(event_bin_counts, tf.float32)
    bin_n = tf.reduce_sum(event_bin_counts, axis=0)
    pbins = bin_n / tf.reduce_sum(bin_n)  # Compute the marginal bin probability

    # Compute the marginal probability of making a correct decision given an
    # observed probability bin.
    tiny = np.finfo(np.float32).tiny
    pcorrect = event_bin_counts[1, :] / (bin_n + tiny)

    # Compute the ECE statistic as defined in reference [1].
    ece = tf.reduce_sum(pbins * tf.abs(pcorrect - pmean_observed))

  return ece

def print_performance(cm):
    tp = np.diagonal(cm).astype(np.float)
    tpfp = np.sum(cm, axis=0).astype(np.float)  # sum of each col
    tpfn = np.sum(cm, axis=1).astype(np.float)  # sum of each row
    acc = np.sum(tp) / np.sum(cm)
    precision = tp / tpfp
    recall = tp / tpfn
    f1 = (2 * precision * recall) / (precision + recall)
    mf1 = np.mean(f1)

    print("Sample: {}".format(np.sum(cm)))
    print("W: {}".format(tpfn[W]))
    print("N1: {}".format(tpfn[N1]))
    print("N2: {}".format(tpfn[N2]))
    print("N3: {}".format(tpfn[N3]))
    print("REM: {}".format(tpfn[REM]))
    print("{},{},{},{},{}".format(tpfn[W],tpfn[N1],tpfn[N2],tpfn[N3],tpfn[REM]))
    print("Confusion matrix:")
    print(cm)
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1: {}".format(f1))
    print("Overall accuracy: {}".format(acc))
    print("Macro-F1: {}".format(mf1))

def perf_overall(data_dir, ensembling):
    # Remove non-output files, and perform ascending sort
    allfiles = os.listdir(data_dir)
    outputfiles = []
    for idx, f in enumerate(allfiles):
        if re.match("^output_.+\d+\.npz", f):
            outputfiles.append(os.path.join(data_dir, f))
    outputfiles.sort()

    y_true = []
    y_pred = []
    prob_pred = []

    for fpath in outputfiles:
        with np.load(fpath, allow_pickle=True) as f:

            f_y_true = np.hstack(f["y_true"])
            f_y_pred = np.hstack(f["y_pred"])
            f_prob_pred = np.vstack(f["prob_pred"]) if ensembling else np.hstack(f["prob_pred"])

            y_true.extend(f_y_true)
            y_pred.extend(f_y_pred)
            prob_pred.extend(f_prob_pred)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    prob_pred = np.reshape(np.hstack(prob_pred), (-1, 5))

    k = cohen_kappa_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="weighted")


    print("\nNetwork Performance - Overall")
    print_performance(cm)
    print("Cohen's Kappa: {}".format(k))
    print("macro-F1 : {}".format(mf1))
    print("weighted-F1: {}".format(f1))

    max_prob_pred = np.amax(prob_pred, axis=1)
    max_prob_pred_mean = max_prob_pred.mean()
    print("confidence: {}".format(max_prob_pred_mean))

    tensor_prob_pred = tf.convert_to_tensor(prob_pred, dtype=np.float32)

    tensor_labels_true = tf.convert_to_tensor(y_true, dtype=np.int32)
    tensor_ECE = expected_calibration_error(20, logits=tensor_prob_pred, labels_true=tensor_labels_true,
                                            labels_predicted=None, name=None)

    with tf.Session() as sess:
        ECE = tensor_ECE.eval()
        print("Expected Calibration Error {}".format(ECE))

def perf_overall_selected_prob(data_dir, ensembling):
    # Remove non-output files, and perform ascending sort
    allfiles = os.listdir(data_dir)
    outputfiles = []
    for idx, f in enumerate(allfiles):
        if re.match("^output_.+\d+\.npz", f):
            outputfiles.append(os.path.join(data_dir, f))
    outputfiles.sort()

    y_true = []
    y_pred = []
    prob_pred = []
    prob_pred_max = []

    query_instances = []
    correct_among_query = []

    for fpath in outputfiles:
        with np.load(fpath, allow_pickle=True) as f:

            f_y_true = f["y_true"]
            f_y_pred = f["y_pred"]
            f_prob_pred = f["prob_pred"]

            for sub_f_idx, each_prediction in enumerate(zip(f_y_true, f_y_pred, f_prob_pred)):
                each_y_true, each_y_pred, each_prob_pred = each_prediction
                if not ensembling:
                    each_prob_pred = np.reshape(each_prob_pred, (-1, 5))

                n_examples = len(each_y_true)

                each_y_true = np.asarray(each_y_true)
                each_y_pred = np.asarray(each_y_pred)
                each_prob_pred = np.asarray(each_prob_pred)
                each_prob_pred_max = np.max(each_prob_pred, axis=1)

                threshold = 5 * 1e-2
                # Probabilities Rule selection
                idx_threshold = int((1 - threshold) * n_examples)

                prob_threshold = np.sort(each_prob_pred_max)[::-1][idx_threshold]
                removed_idx = np.where(each_prob_pred_max <= prob_threshold)[-1]

                query_instances.append(len(removed_idx))
                correct_among_query.append(np.sum(each_y_true[removed_idx] == each_y_pred[removed_idx]))

                y_true.extend(np.delete(each_y_true, removed_idx))
                y_pred.extend(np.delete(each_y_pred, removed_idx))
                prob_pred.extend(np.delete(each_prob_pred, removed_idx, axis=0))
                prob_pred_max.extend(np.delete(each_prob_pred_max, removed_idx))

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    prob_pred = np.reshape(np.hstack(prob_pred), (-1, 5))
    prob_pred_max = np.asarray(prob_pred_max)

    query_instances = np.asarray(query_instances)
    correct_among_query = np.asarray(correct_among_query)

    percentage_of_query = np.sum(query_instances) / len(y_true)
    # percentage_of_misclassified = 1 - (np.mean(correct_among_query / query_instances))
    percentage_of_misclassified = 1 - (np.sum(correct_among_query) / np.sum(query_instances))
    print("percentage_of_query: {}".format(percentage_of_query))
    print("percentage_of_misclassified : {}".format(percentage_of_misclassified))

    k = cohen_kappa_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="weighted")

    print("\nNetwork Performance  - On Selected with {}-fixed-q%".format("max-prob-mean" if ensembling else "max-prob"))
    print_performance(cm)
    print("Cohen's Kappa: {}".format(k))
    print("macro-F1 : {}".format(mf1))
    print("weighted-F1: {}".format(f1))

    max_prob_pred_mean = prob_pred_max.mean()
    print("confidence: {}".format(max_prob_pred_mean))

    tensor_prob_pred = tf.convert_to_tensor(prob_pred, dtype=np.float32)

    tensor_labels_true = tf.convert_to_tensor(y_true, dtype=np.int32)
    tensor_ECE = expected_calibration_error(20, logits = tensor_prob_pred, labels_true = tensor_labels_true,
    labels_predicted = None, name = None)

    with tf.Session() as sess:
        ECE = tensor_ECE.eval()
        print("Expected Calibration Error {}".format(ECE))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="Results",
                        help="Directory where to load prediction outputs")
    args = parser.parse_args()

    ensembling_list = [False, True]

    if args.data_dir is not None:

        for ensembling in ensembling_list:

            data_dir = args.data_dir if not ensembling \
                else os.path.join(args.data_dir, 'MC30')

            print(" ")
            print(f"\nModel: {os.path.basename(args.data_dir)} ; Monte Carlo Dropout: {ensembling}")

            # Compute overall performance
            perf_overall(data_dir=data_dir, ensembling=ensembling)

            ##############################################################################################################
            # Fixed q%=5% number instances selected on all the PSGs
            # Select misclassified and compute overall performance on selected
            # fixed q%=5% with query on max probability values or on mean computed on the N max probability values
            # or ratio of mean computed on the N max probability values
            perf_overall_selected_prob(data_dir=data_dir, ensembling=ensembling)
            ##############################################################################################################


if __name__ == "__main__":
    main()
