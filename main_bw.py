import os
import argparse
import sys
import tensorflow as tf
from tensorflow.python.training import training_util
from utils import *
from models import *
from methods import *
from tensorflow.python import debug as tf_debug
import time


checkpoint_every = 2000
num_checkpoints = 5
print("The checkpint will be saved every " + str(checkpoint_every) + " iterations.")
print("The maximum checkpints saved is: " + str(num_checkpoints))


def configure_learning_rate(args, global_step):
    if args.lr_policy == 'fixed':
        return tf.constant(args.lr, name='fixed_learning_rate')
    elif args.lr_policy == 'inv':
        with tf.variable_scope("InverseTimeDecay"):
            global_step = tf.cast(global_step, tf.float32)
            denom = tf.add(1.0, tf.multiply(args.lr_gamma, global_step))
            return tf.multiply(args.lr, tf.pow(denom, -args.lr_power))
    else:
        raise ValueError('lr_policy [%s] was not recognized',
                         args.lr_policy)


def single_global_step(global_step):
    global_step = tf.cast(global_step, tf.float32)
    return global_step


def main(args):

    # Log
    if args.log_dir:
        if tf.gfile.Exists(args.log_dir):
            tf.gfile.DeleteRecursively(args.log_dir)
        tf.gfile.MakeDirs(args.log_dir)

    # Preprocess
    mean = mean_file_loader('ilsvrc_2012')
    train_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.Normalize(mean),
        transforms.RandomCrop(227),
        transforms.RandomHorizontalFlip()
    ], 'TrainPreprocess')
    test_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.Normalize(mean),
        transforms.CenterCrop(227)
    ], 'TestPreprocess')

    # Datasets
    source_dataset = datasets.CSVImageLabelDataset(args.source)
    target_dataset = datasets.CSVImageLabelDataset(args.target)

    # Loaders
    source, (source_init,) = loader.load_data(
        loader.load_dataset(source_dataset, batch_size=args.batch_size,
                            transforms=(train_transform,)))
    target, (target_train_init, target_test_init) = loader.load_data(
        loader.load_dataset(target_dataset, batch_size=args.batch_size,
                            transforms=(train_transform,)),
        loader.load_dataset(target_dataset, batch_size=args.batch_size,
                            transforms=(test_transform,)))

    # Variables
    training = tf.get_variable('train', initializer=True, trainable=False,
                               collections=[tf.GraphKeys.LOCAL_VARIABLES])

    # Loss weights
    loss_weights = [float(i) for i in args.loss_weights.split(',') if i]

    # Construct base model
    base_model = Alexnet(training, fc=-1, pretrained=True)

    # Prepare input images
    # method = DeepAdaptationNetwork(base_model, 31)
    # method = JointAdaptationNetwork(base_model, 31)
    method = AdversarialJointAdaptationNetwork(base_model, 31)

    # Losses and accuracy
    global_step = training_util.create_global_step()
    loss, accuracy, cross_entropy_loss, jmmd_loss, param_bw = method((source[0], target[0]),
                                                                     (source[1], target[1]),
                                                                     loss_weights,
                                                                    global_step)
    
    # Add output dir for summaries and checkpoints
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))


    with tf.Session() as sess:
        bw_offset = param_bw[0]
        bw_scale = param_bw[1]
        jmmd_loss_neg = tf.negative(jmmd_loss)
    
        # Add summary for loss and accuracy
        tf.summary.scalar('accuracy', accuracy / args.batch_size * 100)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('cross entropy loss', cross_entropy_loss)
        tf.summary.scalar('jmmd loss', jmmd_loss)
        summary_op = tf.summary.merge_all()
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        test_summary_dir = os.path.join(out_dir, "summaries", "test")
        test_summary_writer = tf.summary.FileWriter(test_summary_dir)
    
        # Add Checkpoint directory
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)
    
        # Optimize
        var_list1 = list(filter(lambda x: not x.name.startswith('Linear'),
                                tf.global_variables()))
        var_list2 = list(filter(lambda x: x.name.startswith('Linear'),
                                tf.global_variables()))
        grads = tf.gradients(loss, var_list1 + var_list2)
        learning_rate = configure_learning_rate(args, global_step)
        train_op = tf.group(
            tf.train.MomentumOptimizer(learning_rate, args.momentum)
                .apply_gradients(zip(grads[:len(var_list1)], var_list1)),
            tf.train.MomentumOptimizer(learning_rate * 10, args.momentum)
                .apply_gradients(zip(grads[len(var_list1):], var_list2),
                                 global_step=global_step))
        # added bu yuzeng
        adv_jmmd_loss_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(jmmd_loss_neg, var_list=param_bw)
    
        # Initializer
        init = tf.group(tf.global_variables_initializer(), source_init)
        train_init = tf.group(tf.assign(training, True), target_train_init)
        test_init = tf.group(tf.assign(training, False), target_test_init)
    
        # saver
        #saver = tf.train.Saver()
        #checkpoint_dir = './checkpoint'
    
        # Run Session
        # modified
    
        # save flag
        save_flag_50 = 0
        save_flag_60 = 0
        save_flag_70 = 0
    
        print("Begin Training!!")
        # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.run(init)
        sess.run(train_init)
        print("Train_init finished!!")

        # restore the checkpoint!
        #saver.restore(sess, os.path.join(checkpoint_prefix, '-3'))
        #saver.restore(sess, 'runs/1512927344/checkpoints/model-32000')
        #print("Restore the checkpoint")

        for _ in range(args.max_steps):
            _, summaries, lr_val, loss_val, cross_entropy_loss_val, jmmd_loss_val, accuracy_val, step_val, offset_val, scale_val = \
                sess.run(
                    [train_op, summary_op, learning_rate, loss, cross_entropy_loss, jmmd_loss, accuracy, global_step, bw_offset, bw_scale])
            train_summary_writer.add_summary(summaries, step_val)
                  
            accuracy_val = float(accuracy_val) * 100.0 / 64.0  
            if step_val % args.print_freq == 0:
                print('  step: %d\tlr: %.8f\tloss: %.3f\tce_loss: %.3f\tjmmd_loss: %.3f\taccuracy: %.3f\toffset: %.3f\tscale: %.3f%%' %
                      (step_val, lr_val, loss_val, cross_entropy_loss_val, jmmd_loss_val,
                       float(accuracy_val), offset_val, scale_val))
    
                # saver.save(sess, checkpoint_dir + '/model.ckpt', global_step = step_val)
    
            if step_val % args.test_freq == 0:
                accuracies = []
                sess.run(test_init)
                summaries, step_val = sess.run([summary_op, global_step])
                for _ in range(20):
                    accuracies.append(sess.run(accuracy))
                print('test accuracy: %.3f' % ( float(sum(accuracies)) * 100 / (20*64)))

                test_summary_writer.add_summary(summaries, step_val)
                sess.run(train_init)

            # sess.run for discriminator
            if float(accuracy_val) / 100 > 0.5:
                if save_flag_50 == 0:
                    # saver.save(sess, checkpoint_dir + '/model.ckpt', global_step = tf.train.get_global_step())
                    save_flag_50 = 1
                for i in range(0, 1):
                    _, discriminator_loss = sess.run([adv_jmmd_loss_op, jmmd_loss_neg])
                    # print (' The discriminator loss is: %.3f' % (discriminator_loss))
    
            if (float(accuracy_val) / 100 > 0.6) and (save_flag_60 == 0):
                # saver.save(sess, checkpoint_dir + '/model.ckpt', global_step = tf.train.get_global_step())
                save_flag_60 = 1
    
            if (float(accuracy_val) / 100 > 0.7) and (save_flag_70 == 0):
                # saver.save(sess, checkpoint_dir + '/model.ckpt', global_step = tf.train.get_global_step())
                save_flag_70 = 1

            ## [YueSun's Action] Save the checkpoints every {checkpoint_every} loops.
            if step_val % checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=step_val)
                print("Saved model checkpoint to {}\n".format(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Initial learning rate.')
    parser.add_argument('--lr-policy', type=str, choices=['fixed', 'inv'],
                        default='inv',
                        help='Learning rate decay policy.')
    parser.add_argument('--lr-gamma', type=float, default=2e-3,
                        help='Learning rate decay parameter.')
    parser.add_argument('--lr-power', type=float, default=0.75,
                        help='Learning rate decay parameter.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Weight momentum for the solver.')
    parser.add_argument('--loss-weights', type=str, default='',
                        help='Comma separated list of loss weights.')
    parser.add_argument('--max-steps', type=int, default=50000,
                        help='Number of steps to run trainer.')
    parser.add_argument('--source', type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             'data/office/amazon.csv'),
                        help='Source list file of which every lines are '
                             'space-separated image paths and labels.')
    parser.add_argument('--target', type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             'data/office/webcam.csv'),
                        help='Target list file with same layout of source list '
                             'file. Labels are only used for evaluation.')
    parser.add_argument('--base-model', type=str, choices=['alexnet'],
                        default='alexnet', help='Basic model to use.')
    parser.add_argument('--method', type=str, choices=['DAN'], default='DAN',
                        help='Algorithm to use.')
    parser.add_argument('--sampler', type=str,
                        choices=['none', 'fix', 'random'],
                        default='random',
                        help='Sampler for MMD and JMMD. (valid only when '
                             '--loss=mmd or --lost=jmmd)')
    parser.add_argument('--print-freq', type=int, default=100,
                        help='')
    parser.add_argument('--test-freq', type=int, default=300,
                        help='')
    parser.add_argument('--kernel-mul', type=float, default=2.0,
                        help='Kernel multiplier for MMD and JMMD. (valid only '
                             'when --loss=mmd or --lost=jmmd)')
    parser.add_argument('--kernel-num', type=int, default=5,
                        help='Number of kernel for MMD and JMMD. (valid only '
                             'when --loss=mmd or --lost=jmmd)')
    parser.add_argument('--log-dir', type=str, default='',
                        help='Directory to put the log data.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=lambda _: main(FLAGS), argv=[sys.argv[0]] + unparsed)
