# Copyright 2020 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Local and distributed trainers on TensorFlow backend"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import tensorflow as tf
from EgoGraph_encoder import EgoGraph_Encoder
from config_file.config_PMGT import FLAGS


class LocalTFTrainer(object):
    """Base class of TF trainer

  Args:
    model_func: A model instance.
    epoch: training epochs.
    optimizer: A tf optimizer instance.
  """

    def __init__(self,
                 graph,
                 epoch=100,
                 optimizer=tf.train.AdamOptimizer()):
        if not graph:
            raise NotImplementedError('gl graph to be implemented.')
        print('initialize MGBert model')
        self.graph = graph
        self._epoch = epoch
        self._optimizer = optimizer
        self.sess = None
        self.saver = None
        self.rng = random.Random(FLAGS.seed)
        self.model = EgoGraph_Encoder()

    def __exit__(self, exc_type, exc_value, tracebac):
        if self.sess:
            self.sess.close()
        return True

    def init(self, **kwargs):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.local_variables_initializer())
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=3)

