import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, MaxPooling2D, Flatten, concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from wandb.integration.keras import WandbMetricsLogger
import wandb
import gc
import os

#PHC & img data directory
DATA_DIR = "PHC_data" 
IMG_DIR = "grey_ost_img"

def get_file_path(dim: str, dataset_class: str) -> str:
    
    filename = f"{dim}_{dataset_class}_PHC_data_t195.npy"
    return os.path.join(DATA_DIR, filename)

def load_data(config):
    exp_mode = config.experiment_mode
    data_class = config.dataset_class

    #Labels
    lb_path = os.path.join(IMG_DIR, "balanced_lb.npy")
    lb = np.array(np.load(lb_path, allow_pickle=True), dtype=np.float64)
    
    #Image data conditional
    img_train, img_test = None, None
    if "img" in exp_mode:
        img_path = os.path.join(IMG_DIR, "balanced_img.npy")
        img_data = np.array(np.load(img_path, allow_pickle=True), dtype=np.float64)
        # Split Image
        train_img, test_img, _, _ = train_test_split(img_data, lb, test_size=0.2, random_state=42)
        img_train = np.expand_dims(train_img, axis=-1)
        img_test = np.expand_dims(test_img, axis=-1)

    #Dim 0 data conditional
    ph0_train, ph0_test = None, None
    if "dim0" in exp_mode:
        fpath = get_file_path(0, data_class)
        ph0_data = np.array(np.load(fpath, allow_pickle=True), dtype=np.float64).reshape(1143, 256, 400, 1)     
        train_ph0, test_ph0, _, _ = train_test_split(ph0_data, lb, test_size=0.2, random_state=42)
        ph0_train, ph0_test = train_ph0, test_ph0

    #Dim 1 data conditional
    ph1_train, ph1_test = None, None
    if "dim1" in exp_mode:
        fpath = get_file_path(1, data_class)
        ph1_data = np.array(np.load(fpath, allow_pickle=True), dtype=np.float64).reshape(1143, 256, 400, 1)
        train_ph1, test_ph1, _, _ = train_test_split(ph1_data, lb, test_size=0.2, random_state=42)
        ph1_train, ph1_test = train_ph1, test_ph1
        
    #Split labels
    _, _, y_train, y_test = train_test_split(lb, lb, test_size=0.2, random_state=42)

    #Assemble Inputs List based on Mode
    inputs_train = []
    inputs_test = []
    
    # Order: [Dim0, Dim1, Img] (or subset thereof)
    if "dim0" in exp_mode:
        inputs_train.append(ph0_train)
        inputs_test.append(ph0_test)
    if "dim1" in exp_mode:
        inputs_train.append(ph1_train)
        inputs_test.append(ph1_test)
    if "img" in exp_mode:
        inputs_train.append(img_train)
        inputs_test.append(img_test)
        
    return inputs_train, inputs_test, y_train, y_test

def build_cnn_branch(input_shape, config):
    inp = Input(shape=input_shape)
    x = Conv2D(config.conv1_filter, (3,3), strides=2, activation='relu', kernel_initializer='he_normal', 
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=config.l1, l2=config.l2), padding='same')(inp)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3,3), padding='same')(x)
    x = Dropout(config.dropout)(x)

    x = Conv2D(config.conv2_filter, (3,3), strides=2, activation='relu', kernel_initializer='he_normal', 
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=config.l1, l2=config.l2), padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3,3), padding='same')(x)
    x = Dropout(config.dropout)(x)
    
    flat = Flatten()(x)
    return inp, flat

def train():
    with wandb.init() as run:
        config = run.config
        
        print(f"--- Running: {config.dataset_class} | {config.experiment_mode} | v2 ---") 
        
        try:
            X_train, X_test, y_train, y_test = load_data(config)
        except FileNotFoundError as e:
            print(f"Data missing for this combination: {e}")
            return 

        #Build model framework based on data avaliable  
        input_layers = []
        flattened_outputs = []
        
        for input_data in X_train:
            inp_shape = input_data.shape[1:]
            inp_layer, flat_out = build_cnn_branch(inp_shape, config)
            input_layers.append(inp_layer)
            flattened_outputs.append(flat_out)
        
        if len(flattened_outputs) > 1:
            merged = concatenate(flattened_outputs)
        else:
            merged = flattened_outputs[0]

        x = Dense(config.layer_1, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=config.l1, l2=config.l2), activation='relu')(merged)
        x = Dropout(config.dropout)(x)
        x = Dense(config.layer_2, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=config.l1, l2=config.l2), activation='relu')(x)
        x = Dropout(config.dropout)(x)
        x = Dense(config.layer_3, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=config.l1, l2=config.l2), activation='relu')(x)
        output_layer = Dense(3, activation='softmax')(x)
        
        CNN = Model(inputs=input_layers, outputs=output_layer)
        
        CNN.compile(
            optimizer='adam', 
            loss='categorical_crossentropy', 
            metrics=['accuracy', 
                     keras.metrics.SpecificityAtSensitivity(0.9), 
                     keras.metrics.SensitivityAtSpecificity(0.9),
                     keras.metrics.F1Score(average='macro'), 
                     keras.metrics.Recall(), 
                     keras.metrics.Precision()])
        

        callbacks = [
            WandbMetricsLogger(log_freq=5), 
            EarlyStopping(monitor='val_accuracy', patience=5, verbose=2)
        ]
        
        #Handle single vs list input for fit()
        fit_x_train = X_train[0] if len(X_train) == 1 else X_train
        fit_x_test = X_test[0] if len(X_test) == 1 else X_test
        

        CNN.fit(x=fit_x_train, y=y_train, epochs=50, validation_split=0.125, callbacks=callbacks, verbose=2)

        results = CNN.evaluate(x=fit_x_test, y=y_test, return_dict=True)
        test_metrics = {}
        for name, value in results.items():
            test_metrics[f"test/{name}"] = value

        wandb.log(test_metrics)
        
        #Memory clean up
        del CNN, X_train, X_test, y_train, y_test
        gc.collect()
        tf.keras.backend.clear_session()

if __name__ == "__main__":
    train()
