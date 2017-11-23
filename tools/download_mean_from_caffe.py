import os
import argparse
from tools_utils import download
import pickle


caffe_mean_files = {
    # name: (url, path to file if compression, save path for binaryproto, save path for pkl)
    # From http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz
    'ilsvrc_2012': (
        'https://github.com/BVLC/caffe/raw/master/python/caffe/imagenet/ilsvrc_2012_mean.npy',
        'ilsvrc_2012_mean.npy',
        '../models/ilsvrc_2012_mean.pkl'
    )
}


def extract_npy(proto, output):
    import numpy as np
    data = np.load(proto)
    # from c x h x w => h x w x c, and from BGR -> RGB
    mean = np.flip(data.transpose([1, 2, 0]).astype(np.float32), 2)
    pickle.dump(mean, open(output, 'wb'), protocol=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Caffe mean file and convert it to NumPy pickle')
    parser.add_argument('--proto', type=str, default='', help='download Caffe binaryproto to or load it from this path')
    parser.add_argument('-o', dest='output', type=str, default='', help='save NumPy pickle to this path')
    parser.add_argument('-m', dest='mean', type=str, choices=list(caffe_mean_files.keys()),
                        default='ilsvrc_2012', help='Mean file to download')
    args = parser.parse_args()
    if len(args.proto) == 0:
        args.proto = os.path.join(os.path.dirname(__file__), caffe_mean_files[args.mean][1])
    if len(args.output) == 0:
        args.output = os.path.join(os.path.dirname(__file__), caffe_mean_files[args.mean][2])
    if not os.path.exists(args.proto):
        print('Downloading npy...')
        download(args.proto, caffe_mean_files[args.mean][0])

    print('Extracting mean file...')
    extract_npy(args.proto, args.output)
