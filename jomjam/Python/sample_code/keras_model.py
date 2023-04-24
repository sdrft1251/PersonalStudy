import tensorflow as tf

input_tens = tf.keras.input(shape=(None,None))
x = tf.keras.layers.Conv1D(filters=FILTERNUM, kernel_size=KERNELSIZE)(input_tens)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Dense(52)(x)

model = tf.keras.Model(inputs=input_tens, outputs=x)
