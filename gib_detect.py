#!/usr/bin/python

import pickle
import gib_detect_train

model_data = pickle.load(open('gib_model.pki'))

while True:
    l = raw_input().decode("UTF-8")
    model_mat = model_data['mat']
    threshold = model_data['thresh']
    print gib_detect_train.avg_transition_prob(l, model_mat) > threshold

