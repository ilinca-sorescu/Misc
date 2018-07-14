import tensorflow as tf
import shutil
from random import uniform
import math

CSV_COLS = ['r', 'h', 'V']
LABEL_COL = 'V'
FEATURES = CSV_COLS[0:2]
DEFAULTS = [[0.5], [0.5], [0.4]]
TRAIN = './c-train*'
VALID = './c-valid*'

def make_feature_cols():
    return [tf.feature_column.numeric_column(k) for k in FEATURES]

def read_dataset(filename, mode, batch_size=512):
    def _input_fn():
        def _add_noise(v):
            noise = uniform(0.0, v/10)
            return v+noise
        
        def _decode_line(line):
            columns = tf.decode_csv(line, DEFAULTS)
            features = dict(zip(CSV_COLS, columns))
            for k,v in features.items():
                v = _add_noise(v)
                features[k] = v
            label = features.pop(LABEL_COL)
            return features, label
        
        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None
        else:
            num_epochs = 1
        dataset = tf.data.Dataset.list_files(filename) \
        .flat_map(tf.data.TextLineDataset) \
        .map(_decode_line)
        dataset = dataset.repeat(num_epochs) \
        .batch(batch_size)
        return dataset.make_one_shot_iterator().get_next()
    return _input_fn

def get_train():
    return read_dataset(TRAIN, tf.estimator.ModeKeys.TRAIN)
    
def get_valid():
    return read_dataset(VALID, tf.estimator.ModeKeys.EVAL)
    
OUTDIR = "c"
shutil.rmtree(OUTDIR, ignore_errors = True)

model = tf.estimator.DNNRegressor(hidden_units = [35, 11, 11, 2],
        feature_columns = make_feature_cols(), model_dir = OUTDIR)
model.train(input_fn = get_train(), steps = 1000)
metrics = model.evaluate(input_fn = get_valid())
print('RMSE on validation is {}.'.format(math.sqrt(metrics['average_loss'])))
