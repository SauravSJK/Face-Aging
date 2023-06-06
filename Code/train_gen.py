import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import get_data
import generator
import numpy as np
import discriminator
from glob import glob
import write_tfrecords
import tensorflow as tf
from os.path import exists
import tensorflow_datasets as tfds
from tensorflow.data import AUTOTUNE
from tensorflow.io import decode_jpeg
from tensorflow.io import FixedLenFeature
from layer_utils import CustomEarlyStopping
from tensorflow.data import TFRecordDataset
from tensorflow.io import parse_single_example
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import BackupAndRestore
from alt_model_checkpoint.tensorflow import AltModelCheckpoint

# Returns the format for the tfrecord data
def parse_tfrecord(example):
    feature_description = {
        "image": FixedLenFeature([], tf.string),
        "source_age_group": FixedLenFeature([], tf.int64),
        "target_age_group": FixedLenFeature([], tf.int64)
    }
    example = parse_single_example(example, feature_description)
    example["image"] = decode_jpeg(example["image"], channels=3)
    example["source_age_group"] = tf.one_hot(example["source_age_group"], 14)
    example["target_age_group"] = tf.one_hot(example["target_age_group"], 14)
    return example


# Loads the dataset based on whether we are using it for training or validation
def load_dataset(run_type, job_dir):
    files = sorted(glob(job_dir + "/tfrecords/*"))
    if run_type == "training":
        raw_dataset = TFRecordDataset(files[:-1])
    else:
        raw_dataset = TFRecordDataset(files[-1])
    parsed_dataset = raw_dataset.map(parse_tfrecord)
    return parsed_dataset


# Returns the shuffled and batched dataset
def get_dataset(run_type, job_dir=".."):
    dataset = load_dataset(run_type, job_dir)
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(8)
    return dataset


# Train the generator
def train_generator(job_dir="..", epochs=100, learning_rate=0.0002, patience=20):
    tfrecords_dir = job_dir + "/tfrecords"
    if not exists(tfrecords_dir):
        data = get_data.get_data()
        write_tfrecords.write_tfrecords(data)
    dis = load_model(job_dir + "/checkpoint/discriminator/")
    for layer in dis.layers:
        layer._name = layer.name + str("_dis")
    dis._name = "discriminator"
    gen = generator.generator()
    model_ = generator.define_gan(gen, dis, learning_rate, job_dir)

    # Setup a callback for Tensorboard to store logs
    log_dir = job_dir + "/logs/generator/"
    tensorboard_callback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1)

    # Setup a callback for Model Checkpointing
    checkpoint_dir = job_dir + "/checkpoint/generator/"
    model_checkpoint_callback = AltModelCheckpoint(
        filepath=checkpoint_dir,
        alternate_model=gen,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    # Setup a callback for Backup and Restore
    backup_dir = job_dir + "/backup/generator/"
    backup_restore_callback = BackupAndRestore(
        backup_dir,
        save_freq="epoch",
        delete_checkpoint=True,
        save_before_preemption=False)

    early_stopping_callback = CustomEarlyStopping(
        patience=patience)

    model_.fit(
        get_dataset("training", job_dir),
        verbose=2,
        epochs=epochs,
        validation_data=get_dataset("testing", job_dir),
        callbacks=[
            tensorboard_callback,
            model_checkpoint_callback,
            backup_restore_callback,
            early_stopping_callback])
    return gen
