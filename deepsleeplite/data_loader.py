import os

from deepsleeplite.utils import *

import random

import re


class DataLoader(object):

    def __init__(self, data_dir, n_folds, fold_idx):
        self.data_dir = data_dir
        self.n_folds = n_folds
        self.fold_idx = fold_idx

    def load_SleepEDF_files_cv_baseline(self, version=None, n_files=None):
        # Remove non-mat files, and perform ascending sort
        allfiles = os.listdir(self.data_dir)
        npzfiles = []
        train_files = []
        valid_files = []
        test_files = []
        allfiles.sort()

        allfiles_enum = []
        ID_count = 0
        night_count = 1
        one_night_ID = [13, 36, 52]
        for file in allfiles:
            ID_file = int(file[3:5])
            if ID_file == ID_count:
                allfiles_enum.append(file)
            else:
                ID_enum = ('0'+str(ID_count)) if ID_count<10 else str(ID_count)
                file_enum = (file[:3]+ID_enum+file[5:])
                allfiles_enum.append(file_enum)
            if night_count == 2 or int(file[3:5]) in one_night_ID:
                ID_count += 1
                night_count = 1
            else:
                night_count += 1

        if version == 'v1' or version == 'v1_trim':

            for idx, f in enumerate(allfiles):
                if ".npz" in f:
                    npzfiles.append(os.path.join(self.data_dir, f))
            npzfiles.sort()

            # Sleep-EDF Database - Physionet - SC Healthy
            for idx, f in enumerate(allfiles):
                if self.fold_idx < 10:
                    pattern = re.compile("[a-zA-Z0-9]*0{}[1-9][A-Z]0\.npz$".format(self.fold_idx))
                else:
                    pattern = re.compile("[a-zA-Z0-9]*{}[1-9][A-Z]0\.npz$".format(self.fold_idx))
                if pattern.match(f):
                    test_files.append(os.path.join(self.data_dir, f))

            train_files = list(set(npzfiles) - set(test_files))

            with np.load(os.path.join(os.path.split(self.data_dir)[0], 'data_split_v1.npz'), allow_pickle=True) as f:
                ID_valid = f['valid_files'][self.fold_idx]

            # Sleep-EDF Database - Physionet - SC Healthy
            for idxs in ID_valid:
                if idxs == None: continue
                for idx, f in enumerate(allfiles):
                    if idxs < 10:
                        pattern = re.compile("[a-zA-Z0-9]*0{}[1-9][A-Z]0\.npz$".format(idxs))
                    else:
                        pattern = re.compile("[a-zA-Z0-9]*{}[1-9][A-Z]0\.npz$".format(idxs))
                    if pattern.match(f):
                        valid_files.append(os.path.join(self.data_dir, f))

            train_files = list(set(train_files) - set(valid_files))

        elif version == 'v2' or version == 'v2_trim':

            with np.load(os.path.join(os.path.split(self.data_dir)[0], 'data_split_v2.npz'), allow_pickle=True) as f:
                ID_train = f['train_files'][self.fold_idx]
                ID_valid = f['valid_files'][self.fold_idx]
                ID_test = f['test_files'][self.fold_idx]

            # Sleep-EDF Database - Physionet - SC Healthy
            for idxs in ID_train:
                if idxs == None: continue
                for idx, f in enumerate(allfiles_enum):
                    if idxs < 10:
                        pattern = re.compile("[a-zA-Z0-9]*0{}[1-9][A-Z]0\.npz$".format(idxs))
                    else:
                        pattern = re.compile("[a-zA-Z0-9]*{}[1-9][A-Z]0\.npz$".format(idxs))
                    if pattern.match(f):
                        train_files.append(os.path.join(self.data_dir, allfiles[idx]))

            # Sleep-EDF Database - Physionet - SC Healthy
            for idxs in ID_valid:
                if idxs == None: continue
                for idx, f in enumerate(allfiles_enum):
                    if idxs < 10:
                        pattern = re.compile("[a-zA-Z0-9]*0{}[1-9][A-Z]0\.npz$".format(idxs))
                    else:
                        pattern = re.compile("[a-zA-Z0-9]*{}[1-9][A-Z]0\.npz$".format(idxs))
                    if pattern.match(f):
                        valid_files.append(os.path.join(self.data_dir, allfiles[idx]))

            # Sleep-EDF Database - Physionet - SC Healthy
            for idxs in ID_test:
                if idxs == None: continue
                for idx, f in enumerate(allfiles_enum):
                    if idxs < 10:
                        pattern = re.compile("[a-zA-Z0-9]*0{}[1-9][A-Z]0\.npz$".format(idxs))
                    else:
                        pattern = re.compile("[a-zA-Z0-9]*{}[1-9][A-Z]0\.npz$".format(idxs))
                    if pattern.match(f):
                        test_files.append(os.path.join(self.data_dir, allfiles[idx]))

        train_files.sort()
        valid_files.sort()
        test_files.sort()

        return train_files, valid_files, test_files

    def load_DB_files_baseline(self, version=None):
        # Remove non-mat files, and perform ascending sort
        allfiles = os.listdir(self.data_dir)
        train_files = []
        valid_files = []
        test_files = []
        allfiles.sort()

        with np.load(os.path.join(os.path.split(self.data_dir)[0], 'data_split_{}.npz'.format(version)), allow_pickle=True) as f:
            ID_train = f['train_files'][self.fold_idx].tolist()
            ID_valid = f['valid_files'][self.fold_idx].tolist()
            ID_test = f['test_files'][self.fold_idx].tolist()

        if None in ID_train:
            ID_train = [i for i in ID_train if i!=None]
        if None in ID_valid:
            ID_valid = [i for i in ID_valid if i!=None]
        if None in ID_test:
            ID_test = [i for i in ID_test if i!=None]

        tr_files = np.asarray(allfiles)[ID_train]
        va_files = np.asarray(allfiles)[ID_valid]
        te_files = np.asarray(allfiles)[ID_test]

        for tmp in tr_files:
            train_files.append(os.path.join(self.data_dir, tmp))
        for tmp in va_files:
            valid_files.append(os.path.join(self.data_dir, tmp))
        for tmp in te_files:
            test_files.append(os.path.join(self.data_dir, tmp))

        train_files.sort()
        valid_files.sort()
        test_files.sort()

        return train_files, valid_files, test_files

    def _load_npz_file_SleepEDF(self, npz_file):
        """Load data and labels from a npz file."""
        with np.load(npz_file) as f:
            data = f["x"][:,:,0] # single channel EEG Fpz-Cz
            labels = f["y"]
            sampling_rate = f["fs"]
        return data, labels, sampling_rate

    def _load_npz_file_DB(self, npz_file, conditioned):
        """Load data and labels from a npz file."""
        if conditioned:
            with np.load(npz_file) as f:
                data = f["x_eeg"]
                labels = f["y_eeg"]
                sampling_rate = f["fs_eeg"]
                diagnosis = np.vstack((f["diagnose_1"], f["diagnose_2"], f["diagnose_3"]))
                age = f["age"]
                gender = f["gender"]
                bmi = f["bmi"]
                cnds = np.vstack((diagnosis, age, gender, bmi)).T
        else:
            with np.load(npz_file) as f:
                data = f["x_eeg"][:, :, 1]
                labels = f["y_eeg"]
                sampling_rate = f["fs_eeg"]
                cnds=None

        return data, labels, sampling_rate, cnds

    def _load_npz_list_files(self, npz_files, conditioned=False):
        """Load data and labels from list of npz files."""
        data = []
        labels = []
        cnds = []
        fs = None
        for npz_f in npz_files:
            print("Loading {} ...".format(npz_f))
            # TODO condition on database to be loaded
            tmp_data, tmp_labels, sampling_rate = self._load_npz_file_SleepEDF(npz_f)
            # tmp_data, tmp_labels, sampling_rate, tmp_cnds = self._load_npz_file_DB(npz_f, conditioned)
            if fs is None:
                fs = sampling_rate
            elif fs != sampling_rate:
                raise Exception("Found mismatch in sampling rate.")

            # Reshape the data to match the input of the model - conv2d
            tmp_data = np.squeeze(tmp_data)
            tmp_data = tmp_data[:, :, np.newaxis, np.newaxis]

            # Casting
            tmp_data = tmp_data.astype(np.float32)
            tmp_labels = tmp_labels.astype(np.int32)

            data.append(tmp_data)
            labels.append(tmp_labels)
            # cnds.append(tmp_cnds)

        return data, labels, sampling_rate, cnds

    def load_train_data_sequences(self, input_files, seq_length):

        subject_files = input_files
        subject_files.sort()

        # Load training set
        print("\n========== [Fold-{}] ==========\n".format(self.fold_idx))
        print("Load training set:")
        data_train, label_train, sampling_rate, _ = self._load_npz_list_files(npz_files=subject_files)
        print(" ")

        # Extract sequences of length L=seq_length
        data_train, label_train = get_sequences(
            x=data_train, y=label_train, seq_length=seq_length
        )

        print("Training set: n_psg={}".format(len(data_train)))

        data_train = np.vstack(data_train)
        label_train = np.vstack(label_train)

        print("Number of examples = {}".format(len(data_train)))

        print(" ")

        return data_train, label_train, sampling_rate

    def load_valid_data_sequences(self, input_files, seq_length):

        subject_files = input_files
        subject_files.sort()

        # Load validation set
        print("\n========== [Fold-{}] ==========\n".format(self.fold_idx))
        print("Load validation set:")
        data_val, label_val, sampling_rate, _ = self._load_npz_list_files(npz_files=subject_files)
        print(" ")

        # Extract sequences of length L=seq_length
        data_val, label_val = get_sequences(
            x=data_val, y=label_val, seq_length=seq_length
        )

        print("Validation set: n_psg={}".format(len(data_val)))

        data_val = np.vstack(data_val)
        label_val = np.vstack(label_val)

        print("Number of examples = {}".format(len(data_val)))

        print(" ")

        return data_val, label_val, sampling_rate

    @staticmethod
    def load_data_cv_baseline(data_dir, fold_idx, version):

        """Load training and cross-validation sets data files."""
        allfiles = os.listdir(data_dir)
        allfiles.sort()

        allfiles_enum = []
        ID_count = 0
        night_count = 1
        one_night_ID = [13, 36, 52]
        for file in allfiles:
            ID_file = int(file[3:5])
            if ID_file == ID_count:
                allfiles_enum.append(file)
            else:
                ID_enum = ('0' + str(ID_count)) if ID_count < 10 else str(ID_count)
                file_enum = (file[:3] + ID_enum + file[5:])
                allfiles_enum.append(file_enum)
            if night_count == 2 or int(file[3:5]) in one_night_ID:
                ID_count += 1
                night_count = 1
            else:
                night_count += 1

        with np.load(os.path.join(os.path.split(data_dir)[0], 'data_split_{}.npz'.format(version)),
                     allow_pickle=True) as f:
            ID_test = f['test_files'][fold_idx]

        subject_files = []

        # Sleep-EDF Database - Physionet - SC Healthy
        for idxs in ID_test:
            if idxs == None: continue
            for idx, f in enumerate(allfiles_enum):
                if idxs < 10:
                    pattern = re.compile("[a-zA-Z0-9]*0{}[1-9][A-Z]0\.npz$".format(idxs))
                else:
                    pattern = re.compile("[a-zA-Z0-9]*{}[1-9][A-Z]0\.npz$".format(idxs))
                if pattern.match(f):
                    subject_files.append(os.path.join(data_dir, allfiles[idx]))

        subject_files.sort()

        def _load_npz_file_SleepEDF(npz_file):
            """Load data and labels from a npz file."""
            with np.load(npz_file) as f:
                data = f["x"][:, :, 0]
                labels = f["y"]
                sampling_rate = f["fs"]
            return data, labels, sampling_rate

        def load_npz_list_files(npz_files):
            """Load data and labels from list of npz files."""
            data = []
            labels = []
            fs = None
            for npz_f in npz_files:
                print("Loading {} ...".format(npz_f))
                # TO ADD condition on database to be loaded
                tmp_data, tmp_labels, sampling_rate = _load_npz_file_SleepEDF(npz_f)
                if fs is None:
                    fs = sampling_rate
                elif fs != sampling_rate:
                    raise Exception("Found mismatch in sampling rate.")

                # Reshape the data to match the input of the model - conv2d
                tmp_data = np.squeeze(tmp_data)
                tmp_data = tmp_data[:, :, np.newaxis, np.newaxis]

                # # Reshape the data to match the input of the model - conv1d
                # tmp_data = tmp_data[:, :, np.newaxis]

                # Casting
                tmp_data = tmp_data.astype(np.float32)
                tmp_labels = tmp_labels.astype(np.int32)

                data.append(tmp_data)
                labels.append(tmp_labels)

            return data, labels, sampling_rate

        print("Load data from: {}".format(subject_files))
        data, labels, sampling_rate = load_npz_list_files(subject_files)

        return data, labels, subject_files, sampling_rate

    @staticmethod
    def load_test_data_sequences(input_file, seq_length):

        def _load_npz_file_DB(npz_file, conditioned=False):
            """Load data and labels from a npz file."""
            if conditioned:
                with np.load(npz_file) as f:
                    data = f["x_eeg"]
                    labels = f["y_eeg"]
                    sampling_rate = f["fs_eeg"]
                    diagnosis = np.vstack((f["diagnose_1"], f["diagnose_2"], f["diagnose_3"]))
                    age = f["age"]
                    gender = f["gender"]
                    bmi = f["bmi"]
                    cnds = np.vstack((diagnosis, age, gender, bmi)).T
            else:
                with np.load(npz_file) as f:
                    data = f["x_eeg"]
                    labels = f["y_eeg"]
                    sampling_rate = f["fs_eeg"]
                    cnds = None

            return data, labels, sampling_rate, cnds

        # Load test file
        print("Load test file: {}".format(os.path.basename(input_file)))
        data_test, label_test, sampling_rate, _ = _load_npz_file_DB(npz_file=input_file)
        print(" ")

        # Extract sequences of length L=seq_length
        data_test, label_test = get_sequences(
            x=[data_test], y=[label_test], seq_length=seq_length
        )

        data_test = np.vstack(data_test)
        label_test = np.vstack(label_test)

        print("Number of examples = {}".format(len(data_test)))

        print(" ")

        return data_test, label_test, sampling_rate

