import os
import sys
import argparse
from math import sqrt
from tools_utils import download, check
import numpy as np

caffe_models = {
    'alexnet': (
        'http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy',
        'bvlc_alexnet.npy',
        '../models/caffe_alexnet.pkl',
        256 # First fc input channel size
    )
}


def extract_model(input, output, first_fc_in):
    input_data = np.load(input, encoding='latin1')
    param_dict = input_data.item()
    os.environ['GLOG_minloglevel'] = '2'
    import pickle
    model = {}
    first_conv, first_fc = True, True
    for name in sorted(param_dict):
        params = param_dict[name]
        if name.startswith('fc'):
            model[name + '/weight'] = params[0]
            if len(params) > 1:
                model[name + '/bias'] = params[1]
        elif name.startswith('conv') or name.startswith('res'):
            if first_conv:
                # kernel: w x h x in_c x out_c, since this model is based on BGR, we convert it to RGB
                model[name + '/weight'] = params[0][:, :, ::-1, :] # [a:b:step] start from a to b with step
                first_conv = False
            else:
                model[name + '/weight'] = params[0]
            if len(params) > 1:
                model[name + '/bias'] = params[1]
        elif name.startswith('bn'):
            model[name + '/running_mean'] = params[0]
            model[name + '/running_var'] = params[1]
            # Unknown params[2].data
        elif name.startswith('scale'):
            model[name + '/weight'] = params[0]
            model[name + '/bias'] = params[1]
        else:
            print('Unknown layer: %s  %s' % (name, '  '.join([str(param.shape) for param in params])), file=sys.stderr)
    pickle.dump(model, open(output, 'wb'), protocol=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download trained Caffe model and convert it to NumPy pickle')
    parser.add_argument('--npy', dest='input', type=str, default='', help='input .npy path')
    parser.add_argument('-o', dest='output', type=str, default='', help='save NumPy pickle to this path')
    parser.add_argument('-m', dest='model', type=str, choices=list(caffe_models.keys()),
                        default='alexnet', help='Model to download')
    args = parser.parse_args()
    if len(args.output) == 0:
        args.output = os.path.join(os.path.dirname(__file__), caffe_models[args.model][2])
    if len(args.input) == 0:
        args.input = os.path.join(os.path.dirname(__file__), caffe_models[args.model][1])

    if not os.path.exists(args.input):
        print('Downloading npy...')
        download(args.input, caffe_models[args.model][0])

    print('Extracting model...')
    extract_model(args.input, args.output, caffe_models[args.model][3])
