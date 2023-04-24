import tensorflow as tf

####################################################################################
# VGG 16
####################################################################################
class VGG16(tf.keras.layers.Layer):
    def __init__(self):
        super(VGG16, self).__init__(name="VGG16")
        self.layer_1_1 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation="relu", padding='same')
        self.layer_1_2 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation="relu", padding='same')
        self.layer_1_3 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')
        self.layer_2_1 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation="relu", padding='same')
        self.layer_2_2 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation="relu", padding='same')
        self.layer_2_3 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')
        self.layer_3_1 = tf.keras.layers.Conv1D(filters=256, kernel_size=3, activation="relu", padding='same')
        self.layer_3_2 = tf.keras.layers.Conv1D(filters=256, kernel_size=3, activation="relu", padding='same')
        self.layer_3_3 = tf.keras.layers.Conv1D(filters=256, kernel_size=3, activation="relu", padding='same')
        self.layer_3_4 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')
        self.layer_4_1 = tf.keras.layers.Conv1D(filters=512, kernel_size=3, activation="relu", padding='same')
        self.layer_4_2 = tf.keras.layers.Conv1D(filters=512, kernel_size=3, activation="relu", padding='same')
        self.layer_4_3 = tf.keras.layers.Conv1D(filters=512, kernel_size=3, activation="relu", padding='same')
        self.layer_4_4 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')

    def call(self, x):
        """
        x : input data (shape = batch,T,d)
        """
        x = self.layer_1_1(x)
        x = self.layer_1_2(x)
        x = self.layer_1_3(x)
        x = self.layer_2_1(x)
        x = self.layer_2_2(x)
        x = self.layer_2_3(x)
        x = self.layer_3_1(x)
        x = self.layer_3_2(x)
        x = self.layer_3_3(x)
        x = self.layer_3_4(x)
        x = self.layer_4_1(x)
        x = self.layer_4_2(x)
        x = self.layer_4_3(x)
        x = self.layer_4_4(x)
        return x