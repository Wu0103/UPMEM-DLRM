
from __future__ import absolute_import, division, print_function, unicode_literals

from utils.utils     import cli
from functools import reduce
import operator

from inferenceEngine import inferenceEngine
from accelInferenceEngine import accelInferenceEngine
from loadGenerator   import loadGenerator

from data_generator.dlrm_data_caffe2 import DLRMDataGenerator
import torch

from multiprocessing import Process, Queue
import csv
import sys
import os
import time
import numpy as np

import signal

class DLRMDataLoader:

  def __init__(self, args):
    self.args = args
    self.requestQueue    = Queue(maxsize=1024)
    accelRequestQueue = Queue(maxsize=32)
    pidQueue        = Queue()
    responseQueues  = []
    inferenceEngineReadyQueue = Queue()

    # Create load generator to mimic per-server load
    loadGeneratorReturnQueue = Queue()
    self.DeepRecLoadGenerator = Process( target = loadGenerator,
                        args   = (self.args, self.requestQueue, loadGeneratorReturnQueue, inferenceEngineReadyQueue, pidQueue, accelRequestQueue)
                      )

    self.DeepRecLoadGenerator.start()
    self.datagen = DLRMDataGenerator(args)


  def get_next(self):

    request = self.requestQueue.get()
    if request is None: 
      print("DLRMDataLoader.get_next: no more requests available")
      return
      
    (nbatches, lX, lS_l, lS_i) = self.datagen.generate_input_data()
    (nbatches, lT)             = self.datagen.generate_output_data()

    batch_id   = request.batch_id

    lS_l_curr  = np.transpose(np.array(lS_l[batch_id]))
    lS_l_curr  = np.transpose(np.array(lS_l_curr[:request.batch_size]))

    #make offsets[i][0] = 0
    for l in lS_l_curr:
      l[0] = 0

    lS_i_curr = np.array(lS_i[batch_id])
    lS_i_curr = np.array(lS_i_curr[:][:, :request.batch_size * self.args.num_indices_per_lookup])

    lS_T_curr  = (np.array(lT[batch_id][:request.batch_size]))
    lS_X_curr  = np.array(lX[batch_id][:request.batch_size])
    
    return batch_id, torch.tensor(lS_X_curr), torch.tensor(lS_l_curr, dtype=torch.long), torch.tensor(lS_i_curr, dtype=torch.long), torch.tensor(lS_T_curr) 

  def kill_generator(self):
    self.DeepRecLoadGenerator.kill()



