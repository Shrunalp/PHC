import numpy as np
import keras
import gc
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, MaxPooling2D, Flatten, concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import wandb
wandb.login(key="insert api key herer")
from wandb.integration.keras import WandbMetricsLogger

# Define the sweep configuration
sweep_config = {
    'method': "bayes",
    'metric': {
        'name': 'accuracy',
        'goal': 'maximize',
    },
    'parameters': {
        "conv1_filter": {
            "values": [8, 16, 32]
        },
        "conv2_filter": {
            "values": [8, 16, 32]
        },
        "layer_1": {
            "values": [128, 256, 512]
        },
        "layer_2": {
            "values": [64, 128, 256, 512, 1024]
        },
        "layer_3": {
            "values": [32, 64, 128, 256]
        },
        "dropout": {
            "values": [0.1, 0.2]
        },
        "l1": {
            "values": [0.0001, 0.001, 0.01]
        },
        "l2": {
            "values": [0.0001, 0.001, 0.01, 0.1]
        },
    },
}

sweep_id = wandb.sweep(sweep_config, project="Ost_GCNN_Final")


# Define training protocol for PHC
def train():
    with wandb.init() as run:
        config = run.config
        
        ph_data = np.array(np.load("/scratch/spothago/ost_np/training_data/extended_phc_data.npy", allow_pickle=True), dtype=np.float64)
        img_data = np.array(np.load("/scratch/spothago/ost_np/grey_ost_img/balanced_img.npy", allow_pickle=True), dtype=np.float64)
        lb = np.array(np.load("/scratch/spothago/ost_np/grey_ost_img/balanced_lb.npy", allow_pickle=True), dtype=np.float64)
        
        # Train / Test Split
        train, test, y_train, y_test = train_test_split(img_data, lb, test_size=0.2, random_state=42)
        img_train = np.expand_dims(train, axis=-1)
        img_test = np.expand_dims(test, axis=-1)
        
        train, test, y_train, y_test = train_test_split(ph_data, lb, test_size=0.2, random_state=42)
        homology_test = np.expand_dims(test, axis=-1)
        homology_train = np.expand_dims(train, axis=-1)
        
        # CNN Model 
        image_input =  Input(shape=np.shape(img_train)[1:])
    
        x = Conv2D(config.conv1_filter, (3,3), strides=2, activation='relu', kernel_initializer='he_normal', 
                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=config.l1, l2=config.l2), padding='same') (image_input)
        x = BatchNormalization() (x)
        x = MaxPooling2D((3,3), padding='same') (x)
        x = Dropout(config.dropout) (x)
    
        x = Conv2D(config.conv2_filter, (3,3), strides=2, activation='relu', kernel_initializer='he_normal', 
                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=config.l1, l2=config.l2), padding='same') (x)
        x = BatchNormalization() (x)
        x = MaxPooling2D((3,3), padding='same') (x)
        x = Dropout(config.dropout) (x)
        
        x = Flatten()(x)
        
        ph_input =  Input(shape=np.shape(homology_train)[1:])
    
        y = Conv2D(config.conv1_filter, (3,3), strides=2, activation='relu', kernel_initializer='he_normal', 
                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=config.l1, l2=config.l2), padding='same') (ph_input)
        y = BatchNormalization() (y)
        y = MaxPooling2D((3,3), padding='same') (y)
        y = Dropout(config.dropout) (y)
    
        y = Conv2D(config.conv2_filter, (3,3), strides=2, activation='relu', kernel_initializer='he_normal', 
                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=config.l1, l2=config.l2), padding='same') (y)
        y = BatchNormalization() (y)
        y = MaxPooling2D((3,3), padding='same') (y)
        y = Dropout(config.dropout) (y)
    
        y = Flatten()(y)
        
        xy = concatenate([x,y])
    
        xy = Dense(config.layer_1, activation='relu') (xy)
        xy = Dropout(config.dropout) (xy)
        xy = Dense(config.layer_2, activation='relu') (xy)
        xy = Dropout(config.dropout) (xy)
        xy = Dense(config.layer_3, activation='relu') (xy)
    
        output_layer = Dense(3, activation='softmax') (xy)
        CNN_model = Model(inputs=ph_input, outputs=output_layer)


        # Optimization
        CNN_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', keras.metrics.SpecificityAtSensitivity(0.9), keras.metrics.SensitivityAtSpecificity(0.9),
                                                                                      keras.metrics.F1Score(), keras.metrics.Recall(), keras.metrics.Precision()])
        cb = [WandbMetricsLogger(log_freq=5), EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)]
        
        gpus = tf.config.list_logical_devices('GPU')
        for gpu in gpus:
            with tf.device(gpu):
                train = CNN_model.fit(x=[img_train, homology_train], y=y_train, epochs=50, validation_split= 0.125, callbacks=cb)

        results = CNN_model.evaluate(x=[img_test, homology_test], y=y_test)
        metrics_dict = dict(zip(CNN_model.metrics_names, results))
        wandb.log(metrics_dict)
            
        wandb.log({
        "run_type": "Image Reformat",
        })
        
        del ph_data
        del img_data
        del img_train
        del homology_train
        del img_test
        del homology_test
        del lb
        collected = gc.collect()

def main():
    wandb.agent(sweep_id, function=train, count=500)

if __name__=="__main__":
    main()