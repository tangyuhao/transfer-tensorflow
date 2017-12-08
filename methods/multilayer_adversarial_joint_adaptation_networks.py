import tensorflow as tf
import utils.layers as nn
import core.losses as L
from .base_method import BaseMethod
from six.moves import zip_longest
import numpy as np

class MultilayerAdversarialJointAdaptationNetwork(BaseMethod):
    def __init__(self, base_model, n_class):
        super(MultilayerAdversarialJointAdaptationNetwork, self).__init__()
        self.base_model = base_model
        self.feature_dim = base_model.output_dim
        self.n_class = n_class
        self.fcb = nn.Linear(self.feature_dim, 256)
        self.fc = nn.Linear(256, n_class)
        #self.fcb_res_src = tf.layers.dense()	# fully-connected layer used as the first layer for adversarial network for the source domain
        #self.fcb_res_tgt = tf.layers.dense()	# fully-connected layer used as the first layer for adversarial network for the source domain

    def __call__(self, inputs, labels, loss_weights):
        n_class = self.n_class
        PARAM_INIT = 0.00001
        # loss weight: [cross entropy, jmmd]
        inputs = tf.concat(inputs, axis=0)
        features = self.base_model(inputs)
        features = self.fcb(features)
        logits = self.fc(features)
        source_feature, target_feature = tf.split(features, 2)
        source_logits, target_logits = tf.split(logits, 2)

        #source_feature_stoped = tf.stop_gradient(source_feature)
        #target_feature_stoped = tf.stop_gradient(target_feature)

		# code for adversarial network
        #X = tf.placeholder(tf.float32, shape=[None, 784], name='X') 

        def weight_var(shape, name):
            #return tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(PARAM_INIT))
            return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        
        
        def bias_var(shape, name):
            return tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(0))


        D_src_feature_w1 = weight_var([256, 256], 'D_sf_w1')
        D_src_feature_b1 = bias_var([256], 'D_sf_b1')

        D_src_feature_w2 = weight_var([256, 256], 'D_sf_W2')
        D_src_feature_b2 = bias_var([256], 'D_sf_b2')

        D_tgt_feature_w1 = weight_var([256, 256], 'D_tf_W1')
        D_tgt_feature_b1 = bias_var([256], 'D_tf_b1')

        D_tgt_feature_w2 = weight_var([256, 256], 'D_tf_W2')
        D_tgt_feature_b2 = bias_var([256], 'D_tf_b2')

        D_src_logits_w1 = weight_var([n_class, n_class], 'D_sl_W1')
        D_src_logits_b1 = bias_var([n_class], 'D_sl_b1')

        D_src_logits_w2 = weight_var([n_class, n_class], 'D_sl_W2')
        D_src_logits_b2 = bias_var([n_class], 'D_sl_b2')

        D_tgt_logits_w1 = weight_var([n_class, n_class], 'D_tl_W1')
        D_tgt_logits_b1 = bias_var([n_class], 'D_tl_b1')

        D_tgt_logits_w2 = weight_var([n_class, n_class], 'D_tl_W2')
        D_tgt_logits_b2 = bias_var([n_class], 'D_tl_b2')

        param_D = [D_src_feature_w1, D_src_feature_b1, D_src_feature_w2, D_src_feature_b2, D_tgt_feature_w1, D_tgt_feature_b1, D_tgt_feature_w2, D_tgt_feature_b2, D_src_logits_w1, D_src_logits_b1, D_src_logits_w2, D_src_logits_b2, D_tgt_logits_w1, D_tgt_logits_b1, D_tgt_logits_w2, D_tgt_logits_b2]

        def discriminator_src_feature(x):
            h1 = tf.keras.layers.LeakyReLU(tf.add(tf.matmul(x, D_src_feature_w1), D_src_feature_b1))
            D_out = tf.add(tf.matmul(h1, D_src_feature_w2), D_src_feature_b2)
            return D_out

        def discriminator_tgt_feature(x):
            h1 = tf.keras.layers.LeakyReLU(tf.add(tf.matmul(x, D_tgt_feature_w1), D_tgt_feature_b1))
            D_out = tf.add(tf.matmul(h1, D_tgt_feature_w2), D_tgt_feature_b2)
            return D_out
        
        def discriminator_src_logits(x):
            h1 = tf.keras.layers.LeakyReLU(tf.add(tf.matmul(x, D_src_logits_w1), D_src_logits_b1))
            D_out = tf.add(tf.matmul(h1, D_src_logits_w2), D_src_logits_b2)
            return D_out
        
        def discriminator_tgt_logits(x):
            h1 = tf.keras.layers.LeakyReLU(tf.add(tf.matmul(x, D_tgt_logits_w1), D_tgt_logits_b1))
            D_out = tf.add(tf.matmul(h1, D_tgt_logits_w2), D_tgt_logits_b2)
            return D_out

        source_feature_res = discriminator_src_feature(source_feature)
        target_feature_res = discriminator_tgt_feature(target_feature)
        
        source_logits_res = discriminator_src_logits(source_logits)
        target_logits_res = discriminator_tgt_logits(target_logits)

        source_feature_adv = tf.add(source_feature, source_feature_res)
        target_feature_adv = tf.add(target_feature, target_feature_res)	

        source_logits_adv = tf.add(source_logits, source_logits_res)
        target_logits_adv = tf.add(target_logits, target_logits_res)	
        
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels[0],
                                                                       logits=source_logits,
                                                                       name='xentropy')
        cross_entropy_loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        ## the line below is modified
        jmmd_losses = [
            L.jmmd_loss([source_feature_adv, source_logits_adv], 
                        [target_feature_adv, target_logits_adv]),
        ]
        final_jmmd_loss = sum([w * l if w is not None else l
                    for w, l in zip_longest(loss_weights,
                                            jmmd_losses)])
                                            
        lamb = float(2) / (1+ np.exp(-1*10*global_step)) - float(1)
        loss = lamb*final_jmmd_loss + cross_entropy_loss  ## it seems that there should be a lambda here!!!
        correct = tf.nn.in_top_k(target_logits, labels[1], 1)
        #print('target logits: %\t label: %' % ( tf.cast(tf.reduce_max(target_logits), float), tf.cast(tf.reduce_max(labels[1]), float)) )
        accuracy = tf.reduce_sum(tf.cast(correct, tf.int32))
        return loss, accuracy, cross_entropy_loss, final_jmmd_loss, param_D, tf.reduce_max(target_logits), tf.reduce_mean(labels[1])

__all__ = [
    'MultilayerAdversarialJointAdaptationNetwork'
]
