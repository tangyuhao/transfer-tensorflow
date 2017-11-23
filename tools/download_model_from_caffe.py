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
    for name in param_dict:
        params = param_dict[name]
        if name.startswith('fc'):
            if first_fc: # if fc: kernel should be : in_c * out_c
                # caffe kernel: out_c * in_c * h * w, tensorflow layer: h * w * in_c * out_c

                # first turn kernel back to 4D kernel with caffe's order,
                # then transpose to tensorflow's order and flatten
                shape = (params[0].shape[0], first_fc_in,
                         *(int(sqrt(params[0].shape[1] // first_fc_in) + 0.5),) * 2)
                model[name + '/weight'] = params[0].reshape(*shape).transpose([2, 3, 1, 0]).reshape(-1, shape[0])
                print(model[name+'/weight'].shape)
                first_fc = False
            else:
                model[name + '/weight'] = params[0].transpose([1, 0])
            if len(params) > 1:
                model[name + '/bias'] = params[1]
        elif name.startswith('conv') or name.startswith('res'):
            if first_conv:
                model[name + '/weight'] = params[0][:, ::-1].transpose([2, 3, 1, 0])
                first_conv = False
            else:
                model[name + '/weight'] = params[0].transpose([2, 3, 1, 0])
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
