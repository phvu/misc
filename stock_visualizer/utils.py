from __future__ import print_function

import os
import glob
from six.moves import urllib
import tarfile
import shutil

import numpy as np
import pandas as pd


def get_special_symbols(type, dims):
    return -1 * np.ones(dims)


def check_download(data_dir='./data'):
    file_url = 'https://s3.amazonaws.com/ada-demo-data/quandl/processed/nasdaq_transform.tar.gz'
    file_name = 'nasdaq_transform.tar.gz'
    file_path = os.path.join(data_dir, file_name)

    if not os.path.exists(file_path):
        print("Downloading %s to %s" % (file_url, file_path))
        print('If this takes too long, go to the above URL to download the file and put it at {}'.format(file_path))
        filepath, _ = urllib.request.urlretrieve(file_url, file_path)
        statinfo = os.stat(filepath)
        print("Successfully downloaded", file_name, statinfo.st_size, "bytes")

    extracted_dir = os.path.join(data_dir, 'transform/')
    if os.path.exists(extracted_dir):
        shutil.rmtree(extracted_dir)
    with tarfile.open(file_path) as tar:
        tar.extractall(path=data_dir)
    return extracted_dir


def normalize_data(data_dir='./data', sequence_len=40):

    input_dir = check_download(data_dir)

    # do this the easy way, since we don't have too big data
    files = sorted(glob.glob(os.path.join(input_dir, '*.csv')))
    names = [os.path.splitext(os.path.split(x)[1])[0] for x in files]

    all_data = []
    for f in files:
        df = pd.read_csv(f)
        data = df.iloc[:, range(1, len(df.columns) - 2)].values
        all_data.append(data)

    item_count = sum(x.shape[0] for x in all_data)
    all_mean = None
    for data in all_data:
        if all_mean is None:
            all_mean = data.sum(0) / item_count
        else:
            all_mean += (data.sum(0) / item_count)

    all_stddev = None
    for data in all_data:
        if all_stddev is None:
            all_stddev = np.square(data - all_mean).sum(0) / item_count
        else:
            all_stddev += (np.square(data - all_mean).sum(0) / item_count)
    all_stddev = np.sqrt(all_stddev)

    all_data = map(lambda _: (_ - all_mean) / all_stddev, all_data)

    np.savez(os.path.join(data_dir, 'sequences.npz'), **dict(zip(names, all_data)))
    np.savez(os.path.join(data_dir, 'sequences_stats.npz'), mean=all_mean, stddev=all_stddev)

    val_idx = np.random.choice(range(len(all_data)), [len(all_data) * 0.3], replace=False)
    train_idx = [x for x in range(len(all_data)) if x not in val_idx]

    # super lazy to make up a name...
    def abc(idx, out_file):
        arr = []
        labels = []
        for i in idx:
            mat = all_data[i]
            for j in range(0, mat.shape[0], sequence_len):
                sub_mat = mat[min(j, mat.shape[0] - sequence_len):min(j+sequence_len, mat.shape[0]), :]
                arr.append(sub_mat)
                labels.append('{0}_{1:04d}'.format(names[i], j))
        arr = np.dstack(arr)
        np.savez(os.path.join(data_dir, out_file), data=arr, labels=labels)
        print(arr.shape, len(labels))

    abc(train_idx, 'sequences_train.npz')
    abc(val_idx, 'sequences_val.npz')
    abc(range(len(all_data)), 'sequences_all.npz')


def read_data(buckets, data_dir='./data', sequence_len=40):

    data_files = [os.path.join(data_dir, x) for x in ('sequences_train.npz', 'sequences_val.npz')]
    if not all(os.path.exists(_) for _ in data_files):
        normalize_data(data_dir, sequence_len)

    def wtf(f):
        data = np.load(f)['data']
        data_set = [[] for _ in buckets]

        print('Reading {}'.format(f))

        for bucket_id, (source_size, target_size) in enumerate(buckets):
            if data.shape[0] < min(source_size, target_size):
                for d in range(0, data.shape[2]):
                    # some magic here: translate from (0, N-1) to (1, N)
                    source = data[:-1, :, d]
                    target = np.vstack((data[1:, :, d], np.zeros((1, data.shape[1]))))
                    data_set[bucket_id].append([source, target])
                break
        print('Done reading {}'.format(f))
        return data_set

    return wtf(data_files[0]), wtf(data_files[1])
