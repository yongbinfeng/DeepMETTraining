#!/usr/bin/env python
# coding: utf-8

# modified from Jan and Markus's code

import os
import pathlib
import datetime
import h5py
import optparse
import numpy as np

from sklearn.model_selection import train_test_split
import keras
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Flatten, Reshape, Dense, BatchNormalization, Concatenate, Embedding
from keras import optimizers, initializers
from keras.layers import Lambda
from keras.backend import slice

import tensorflow as tf
import keras.backend as K

from tensorflow import train

# Local imports
from cyclical_learning_rate import CyclicLR
from weighted_sum_layer import weighted_sum_layer
from utils import plot_history, get_features, get_targets, convertXY2PtPhi, MakePlots
from loss import custom_loss

def create_model(n_features=4, n_features_cat=2, n_dense_layers=3, activation='tanh', with_bias=False):
    # continuous features
    # ['PF_px', 'PF_py', 'PF_eta', 'PF_puppiWeight']
    inputs_cont = Input(shape=(maxCands, n_features), name='input')
    pxpy = Lambda(lambda x: slice(x, (0, 0, n_features-2), (-1, -1, -1)))(inputs_cont)

    embeddings = []
    for i_emb in range(n_features_cat):
        input_cat = Input(shape=(maxCands, 1), name='input_cat{}'.format(i_emb))
        if i_emb == 0:
            inputs = [inputs_cont, input_cat]
        else:
            inputs.append(input_cat)
        embedding = Embedding(input_dim=emb_input_dim[i_emb], output_dim=emb_out_dim, embeddings_initializer=initializers.RandomNormal(mean=0., stddev=0.4/emb_out_dim), name='embedding{}'.format(i_emb))(input_cat)
        embedding = Reshape((maxCands, 8))(embedding)
        embeddings.append(embedding)

    x = Concatenate()([inputs[0]] + [emb for emb in embeddings])

    for i_dense in range(n_dense_layers):
        x = Dense(8*2**(n_dense_layers-i_dense), activation=activation, kernel_initializer='lecun_uniform')(x)
        x = BatchNormalization(momentum=0.95)(x)

    # List of weights. Increase to 3 when operating with biases
    # Expect typical weights to not be of order 1 but somewhat smaller, so apply explicit scaling
    x = Dense(3 if with_bias else 1, activation='linear', kernel_initializer=initializers.VarianceScaling(scale=0.02))(x)
    #print('Shape of last dense layer', x.shape)

    x = Concatenate()([x, pxpy])
    x = weighted_sum_layer(with_bias, name = "weighted_sum" if with_bias else "output")(x)

    if with_bias:
        x = Dense(2, activation='linear', name='output')(x)

    outputs = x 
    return inputs, outputs


# configuration
usage = 'usage: %prog [options]'
parser = optparse.OptionParser(usage)
parser.add_option('-l', '--load', dest='load',
                  help='load model from timestamp', default='', type='string')
parser.add_option('--withbias', dest='withbias',
                  help='include bias term in the DNN', default=False, action="store_true")
(opt, args) = parser.parse_args()

# general setup
maxCands = 100
epochs = 20
batch_size = 64
emb_out_dim = 8
encoding_charge = {-1: 0, 0: 1, 1: 2}
encoding_pdgId = {-211: 0, -13: 1, -11: 2, 0: 3, 11: 4, 13: 5, 22: 6, 130: 7, 211: 8}

features_cands = ['L1PuppiCands_eta', 'L1PuppiCands_puppiWeight','L1PuppiCands_pt', 'L1PuppiCands_phi',
                         'L1PuppiCands_charge','L1PuppiCands_pdgId']
targets = ['genMet_pt', 'genMet_phi']

##
## read input
##
inputfile = "/eos/cms/store/user/yofeng/L1METML/input_MET_PupCandi.h5"
target_array = get_targets(inputfile, targets)
features_cands_array = get_features(inputfile, features_cands, maxCands)

## remove events with zero met
selections = target_array[:,0] > 10.0
target_array = target_array[selections]
features_cands_array = features_cands_array[selections]

print(target_array)
print(features_cands_array)
print("loaded the data into memory")

##
## preprocessing
##
nevents = target_array.shape[0]
ntargets = target_array.shape[1]
ncandfeatures = features_cands_array.shape[2]
# convert (pt, phi) to x,y
target_array_xy = np.zeros((nevents, ntargets))
target_array_xy[:,0] = target_array[:,0] * np.cos(target_array[:,1])
target_array_xy[:,1] = target_array[:,0] * np.sin(target_array[:,1])
# preprocessing input features
features_cands_array_xy = np.zeros((nevents, maxCands, ncandfeatures))
features_cands_array_xy[:,:,0] = features_cands_array[:,:,0] # eta
features_cands_array_xy[:,:,1] = features_cands_array[:,:,1] # puppiWeight
features_cands_array_xy[:,:,2] = features_cands_array[:,:,2] * np.cos(features_cands_array[:,:,3]) # px
features_cands_array_xy[:,:,3] = features_cands_array[:,:,2] * np.sin(features_cands_array[:,:,3]) # py
features_cands_array_xy[:,:,4] = np.vectorize(encoding_charge.get)(features_cands_array[:,:,4]) # charge
features_cands_array_xy[:,:,5] = np.vectorize(encoding_pdgId.get)(features_cands_array[:,:,5]) # pdgId
print("finished preprocessing")

Xi = features_cands_array_xy[:,:,0:4]
Xc1 = features_cands_array_xy[:,:,4:5]
Xc2 = features_cands_array_xy[:,:,5:]
print("Xi", Xi)
print("Xc1 ", Xc1)
print("Xc2 ", Xc2)
Xc = [Xc1, Xc2]
emb_input_dim = {
    i:int(np.max(Xc[i][0:1000])) + 1 for i in range(len(Xc))
}
print('Embedding input dimensions', emb_input_dim)

# prepare training/val data
Yr = target_array_xy
Xr = [Xi] + Xc
indices = np.array([i for i in range(len(Yr))])
indices_train, indices_test = train_test_split(indices, test_size=0.2, random_state=7)

Xr_train = [x[indices_train] for x in Xr]
Xr_test = [x[indices_test] for x in Xr]
Yr_train = Yr[indices_train]
Yr_test = Yr[indices_test]

# inputs, outputs = create_output_graph()
inputs, outputs = create_model(n_features=4, n_features_cat=len(Xc), with_bias=opt.withbias)

lr_scale = 1.
clr = CyclicLR(base_lr=0.0003*lr_scale, max_lr=0.001*lr_scale, step_size=len(Yr)/batch_size, mode='triangular2')

# create the model
model = Model(inputs=inputs, outputs=outputs)
optimizer = optimizers.Adam(lr=1., clipnorm=1.)
model.compile(loss=custom_loss, optimizer=optimizer, 
               metrics=['mean_absolute_error', 'mean_squared_error'])
#model.compile(loss='mae', optimizer=optimizer,
#               metrics=['mean_absolute_error', 'mean_squared_error'])
model.summary()

if opt.load:
    timestamp = opt.load
else:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
path = f'models/{timestamp}'
pathlib.Path(path).mkdir(parents=True, exist_ok=True)

plot_model(model, to_file=f'{path}/model.png', show_shapes=True)

if opt.load:
    model.load_weights(f'{path}/model.h5')
    print(f'Restored model {timestamp}')

with open(f'{path}/summary.txt', 'w') as txtfile:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: txtfile.write(x + '\n'))

# early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

csv_logger = CSVLogger(f"{path}/loss_history.csv")

# model checkpoint callback
# this saves our model architecture + parameters into model.h5
model_checkpoint = ModelCheckpoint(f'{path}/model_best.h5', monitor='val_loss',
                                   verbose=0, save_best_only=True,
                                   save_weights_only=False, mode='auto',
                                   period=1)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=4, min_lr=0.000001, cooldown=3, verbose=1)

stop_on_nan = keras.callbacks.TerminateOnNaN()


# Run training
history = model.fit(Xr_train, 
                    Yr_train,
                    epochs=epochs,
                    verbose=1,  # switch to 1 for more verbosity
                    validation_data=(Xr_test, Yr_test),
                    callbacks=[early_stopping, clr, stop_on_nan, csv_logger, model_checkpoint],#, reduce_lr], #, lr,   reduce_lr],
                   )

# Plot loss
plot_history(history, path)

model.save(f'{path}/model.h5')
from tensorflow import saved_model
saved_model.simple_save(K.get_session(), f'{path}/saved_model', inputs={t.name:t for t in model.input}, outputs={t.name:t for t in model.outputs})

print("runing predictions on the validation datasets")
# validate the performance
model.load_weights(f'{path}/model_best.h5')
Yr_predict = model.predict(Xr_test)

# baseline 
test_events = Xr_test[0].shape[0]
baseline_xy = np.zeros((test_events, 2))
baseline_xy[:,0] = np.sum(Xr_test[0][:,:,2]*Xr_test[0][:,:,1], axis=1) * (-1)
baseline_xy[:,1] = np.sum(Xr_test[0][:,:,3]*Xr_test[0][:,:,1], axis=1) * (-1)

MakePlots(Yr_test, Yr_predict, baseline_xy)
