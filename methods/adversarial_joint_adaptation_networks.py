import tensorflow as tf
import utils.layers as nn
import core.losses as L
from .base_method import BaseMethod
from six.moves import zip_longest


class AdversarialJointAdaptationNetwork(BaseMethod):
    def __init__(self, base_model, n_class):
        super(AdversarialJointAdaptationNetwork, self).__init__()
        self.base_model = base_model
        self.feature_dim = base_model.output_dim
        self.n_class = n_class
        self.fcb = nn.Linear(self.feature_dim, 256)
        self.fc = nn.Linear(256, n_class)

    def __call__(self, inputs, labels, loss_weights, global_step):
        PARAM_INIT = 0.00001
        # loss weight: [cross entropy, jmmd]
        inputs = tf.concat(inputs, axis=0)
        features = self.base_model(inputs)
        features = self.fcb(features)
        logits = self.fc(features)
        source_feature, target_feature = tf.split(features, 2)
        source_logits, target_logits = tf.split(logits, 2)

        # source_feature_stoped = tf.stop_gradient(source_feature)
        # target_feature_stoped = tf.stop_gradient(target_feature)

        # code for adversarial network
        # X = tf.placeholder(tf.float32, shape=[None, 784], name='X')


        bw_offset = tf.get_variable(name='bw_offset', shape=[1], initializer=tf.constant_initializer(0))
        bw_scale = tf.get_variable(name='bw_scale', shape=[1], initializer=tf.constant_initializer(1))

        param_bw = [bw_offset, bw_scale]


        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels[0],
                                                                       logits=source_logits,
                                                                       name='xentropy')
        cross_entropy_loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        ## the line below is modified
        jmmd_losses = [
            L.jmmd_loss([source_feature, source_logits],
                        [target_feature, target_logits], offset=bw_offset, scale=bw_scale),
        ]
        final_jmmd_loss = sum([w * l if w is not None else l
                               for w, l in zip_longest(loss_weights,
                                                       jmmd_losses)])
        
        a = tf.to_float(global_step)
        lamb = 2.0 / (1 + tf.exp(-10 * a)) - 1.0
        loss = lamb * final_jmmd_loss + cross_entropy_loss 
        loss = final_jmmd_loss + cross_entropy_loss
        correct = tf.nn.in_top_k(target_logits, labels[1], 1)
        accuracy = tf.reduce_sum(tf.cast(correct, tf.int32))
        return loss, accuracy, cross_entropy_loss, final_jmmd_loss, param_bw


__all__ = [
    'AdversarialJointAdaptationNetwork'
]
