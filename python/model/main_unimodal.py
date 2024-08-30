import numpy as np
import tensorflow as tf
import pickle
import random
import tensorflow.keras.backend as K
from training import config as cfg, dice_loss
from training.utils import ENCODER, DECODER, TRANSFORMER
from training.data_generator import DataGenerator2D_CTP_Unimodal
from training.lr_scheduler import StepDecay
from tensorflow.keras import Model, optimizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Concatenate, Input, Lambda
from tensorflow.keras.callbacks import LearningRateScheduler


# -----------------------------------
#            PARAMETERS
# -----------------------------------

# Set seeds for reproducible results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# Print config parameters
print('TRANSFORMER PARAMETERS')
for x in cfg.transformer_params:
    print(x, ':', cfg.transformer_params[x])
print('\nTRAINING PARAMETERS')
for x in cfg.train_params:
    print(x, ':', cfg.train_params[x])
print('\nDATA GENERATOR PARAMETERS')
for x in cfg.params:
    print(x, ':', cfg.params[x])


# -----------------------------------
#            IMPORT DATA
# -----------------------------------

# Importing patient dictionary
print("[INFO] loading patient dictionary...")
with open(cfg.params['dictFile'], 'rb') as output:
    partition = pickle.load(output)

# Calling training generator
train_generator = DataGenerator2D_CTP_Unimodal(partition['training'], shuffle=True, **cfg.params)
test_generator = DataGenerator2D_CTP_Unimodal(partition['testing'], shuffle=False, **cfg.params)


# -----------------------------------
#            BUILD MODEL
# -----------------------------------

# ------ (A.1) IMAGE ENCODER -------
base_network = ENCODER.build(reg=l2(0.00005), shape=cfg.params['dim'])
absolute_diff = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))  # compute the absolute difference between tensors

# Create CTP encoders
inputs, outputs, skip = [], [], []
for w in range(cfg.params['timepoints']):
    # Create inputs and get model outputs
    i = Input(shape=(*cfg.params['dim'], 1))
    o = base_network(i)
    # Append results
    inputs.append(i)
    outputs.append(o[0])
    skip.append(o[1:])

# Concatenate latent vectors and skip connections
encoders = Concatenate(axis=1)(outputs)
connections = []
for i in range(len(skip[0])):
    subtract = []
    for j in range(cfg.params['timepoints']):
        subtract = skip[j][i] if j == 0 else absolute_diff([subtract, skip[j][i]])
    connections.append(subtract)


# -------- (B) MULTIMODAL FUSION --------
# Self-attention: modeling temporal and clinical information via transformers
self_imaging = TRANSFORMER.build(encoders, **cfg.transformer_params, positional_encoding=True)


# ------- (C) OUTCOME PREDICTION ---------
# Lesion prediction: Get decoder output & build model
lesion_prediction = DECODER.build(inputTensor=self_imaging, down=connections, reg=l2(0.00005), n_classes=cfg.params['n_classes'])
model = Model(inputs, lesion_prediction)

# model.summary()  # print model summary
# tf.keras.utils.plot_model(mRS_model, show_shapes=True, rankdir="LR")  # plot model


# -----------------------------------
#            TRAIN MODEL
# -----------------------------------

# Define loss function
dice_func = dice_loss.Dice(nb_labels=cfg.params['n_classes']).loss
mean_dice = dice_loss.Dice(nb_labels=cfg.params['n_classes']).mean_dice

# Callbacks
# LR scheduler
print("[INFO] using 'step-based' learning rate decay...")
sch = StepDecay(initAlpha=cfg.train_params['learning_rate'], factor=0.25, dropEvery=15)  # TODO: try Cosine LR scheduler + WarmUp
# Early Stopping
callback_ES = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)


# Define optimizer
adam_optimizer = optimizers.Adam()

# Compile model
model.compile(loss=dice_func, optimizer=adam_optimizer, metrics=[mean_dice])

"""# Train model
print("[INFO] training network for {} epochs...".format(cfg.train_params['n_epochs']))
H = model.fit(x=train_generator,
              validation_data=test_generator,
              epochs=cfg.train_params['n_epochs'],
              callbacks=[LearningRateScheduler(sch)])  # callback_ES

# Save model weights
# model.save_weights(cfg.params['resultsPath'] + 'ISLES24_2D_model_weights_unimodal')
# print("[INFO] model saved to disk")

# Save training history
# np.save(cfg.params['resultsPath'] + f'ISLES24_2D_trainHistoryDic_unimodal_{extra_name}.npy', H.history)


# -----------------------------------
#          EVALUATE MODEL
# -----------------------------------

predictions = model.predict(test_generator, steps=int(len(partition['testing'])/cfg.params['batch_size']))
np.savez_compressed(cfg.params['resultsPath'] + f'{extra_name}_predictions_unimodal', pred=predictions)"""
