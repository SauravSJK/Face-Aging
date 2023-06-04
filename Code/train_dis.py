import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import get_data
import discriminator
from glob import glob
import write_tfrecords
import tensorflow as tf
from os.path import exists
from tensorflow.data import AUTOTUNE
from tensorflow.io import decode_jpeg
from tensorflow.io import FixedLenFeature
from tensorflow.data import TFRecordDataset
from tensorflow.io import parse_single_example
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import BackupAndRestore


def parse_tfrecord(example):
    feature_description = {
        "image": FixedLenFeature([], tf.string),
        "source_age_group": FixedLenFeature([], tf.int64)
    }
    example = parse_single_example(example, feature_description)
    example["image"] = decode_jpeg(example["image"], channels=3)
    return example["image"], example["source_age_group"]

def load_dataset(run_type, job_dir):
    files = sorted(glob(job_dir + "/tfrecords/*"))
    if run_type == "training":
        raw_dataset = TFRecordDataset(files[:-1])
    else:
        raw_dataset = TFRecordDataset(files[-1])
    parsed_dataset = raw_dataset.map(parse_tfrecord)
    return parsed_dataset

# add repeat
def get_dataset(run_type, job_dir=".."):
    dataset = load_dataset(run_type, job_dir)
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(32)
    return dataset


def train_discriminator(job_dir="..", epochs=100, learning_rate=0.0002, patience=20):
    tfrecords_dir = job_dir + "/tfrecords"
    if not exists(tfrecords_dir):
        data = get_data.get_data()
        write_tfrecords.write_tfrecords(data)
    dis = discriminator.discriminator(learning_rate, job_dir)

    # Setup a callback for Tensorboard to store logs
    log_dir = job_dir + "/logs/discriminator/"
    tensorboard_callback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1)

    # Setup a callback for Model Checkpointing
    checkpoint_dir = job_dir + "/checkpoint/discriminator/"
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_dir,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    # Setup a callback for Backup and Restore
    backup_dir = job_dir + "/backup/discriminator/"
    backup_restore_callback = BackupAndRestore(
        backup_dir,
        save_freq="epoch",
        delete_checkpoint=True,
        save_before_preemption=False)

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True)

    dis.fit(get_dataset("training"),
            epochs=epochs,
            verbose=2,
            callbacks=[
                tensorboard_callback,
                model_checkpoint_callback,
                backup_restore_callback,
                early_stopping_callback],
            validation_data=get_dataset("testing"))


if __name__ == 'main':
    train_discriminator()
