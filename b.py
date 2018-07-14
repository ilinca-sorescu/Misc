import tensorflow as tf
import pandas as pd
import shutil
import math

CSV_COLUMNS = ['r', 'h', 'V']
FEATURES = CSV_COLUMNS[0:2]
LABEL = CSV_COLUMNS[2]

df_train = pd.read_csv('./training_cylinder.csv', header = None, names = CSV_COLUMNS)
df_valid = pd.read_csv('./validation_cylinder.csv', header = None, names = CSV_COLUMNS)

OUTDIR = "cylinder"
shutil.rmtree(OUTDIR, ignore_errors = True)

def make_feature_cols():
    return [tf.feature_column.numeric_column(k) for k in FEATURES]

def make_input_fn(df, num_epochs):
    return tf.estimator.inputs.pandas_input_fn(
            x = df, 
            y = df[LABEL],
            num_epochs = num_epochs,
            shuffle = True)

def make_prediction_input_fn(df, num_epochs):
    return tf.estimator.inputs.pandas_input_fn(
            x = df, 
            y = None,
            num_epochs = num_epochs,
            shuffle = True)

model = tf.estimator.DNNRegressor(hidden_units = [35, 11, 2],
        feature_columns = make_feature_cols(), model_dir = OUTDIR)
model.train(input_fn = make_input_fn(df_train, num_epochs = 100))
metrics = model.evaluate(input_fn = make_input_fn(df_valid, 1))
print('RMSE on validation is {}.'.format(math.sqrt(metrics['average_loss'])))
predictions = model.predict(input_fn = make_prediction_input_fn(df_valid, 1))
for i in range(5):
    print("predicted: {}, real: {}\n".format(predictions.__next__(), df_valid.iloc[i][LABEL]))
