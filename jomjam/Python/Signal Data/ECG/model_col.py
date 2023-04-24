import tensorflow as tf

def return_v01_model():
    input_tens = tf.keras.Input(shape=(187,3,1))
    x = tf.keras.layers.Conv2D(256, kernel_size=(10,3), strides=(5,1),padding='valid')(input_tens)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=(5,1), padding='valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=(5,1), padding='valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=(5,1), padding='valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(5,1), padding='valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(5, activation="softmax")(x)
    model = tf.keras.Model(inputs=input_tens, outputs=x)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
    print(model.summary())
    return model

def return_v02_model():
    input_tens = tf.keras.Input(shape=(187,2,1))
    x = tf.keras.layers.Conv2D(256, kernel_size=(10,2), strides=(5,1),padding='valid')(input_tens)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=(5,1), padding='valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=(5,1), padding='valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=(5,1), padding='valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(5,1), padding='valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(5, activation="softmax")(x)
    model = tf.keras.Model(inputs=input_tens, outputs=x)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
    print(model.summary())
    return model

def return_v03_model():
    input_tens = tf.keras.Input(shape=(187,2,1))
    x = tf.keras.layers.Conv2D(256, kernel_size=(10,2), strides=(5,1),padding='valid')(input_tens)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=(5,1), padding='valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=(5,1), padding='valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Reshape((x.shape[1], x.shape[3]))(x)
    x = tf.keras.layers.LSTM(64)(x)
    x = tf.keras.layers.Dense(5, activation="softmax")(x)
    model = tf.keras.Model(inputs=input_tens, outputs=x)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
    print(model.summary())
    return model

def identity_block(input_tensor, kernel_size, filters, stage, block):
  filters1, filters2, filters3 = filters
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'
  x = tf.keras.layers.Conv2D(filters1, (1, 1),kernel_initializer='he_normal',name=conv_name_base + '2a')(input_tensor)
  x = tf.keras.layers.BatchNormalization(name=bn_name_base + '2a')(x)
  x = tf.keras.layers.Activation('relu')(x)

  x = tf.keras.layers.Conv2D(filters2, (kernel_size,1),padding='same',kernel_initializer='he_normal',name=conv_name_base + '2b')(x)
  x = tf.keras.layers.BatchNormalization(name=bn_name_base + '2b')(x)
  x = tf.keras.layers.Activation('relu')(x)

  x = tf.keras.layers.Conv2D(filters3, (1, 1),kernel_initializer='he_normal',name=conv_name_base + '2c')(x)
  x = tf.keras.layers.BatchNormalization(name=bn_name_base + '2c')(x)

  x = tf.keras.layers.add([x, input_tensor])
  x = tf.keras.layers.Activation('relu')(x)
  return x

def conv_block(input_tensor,kernel_size,filters,stage,block,strides=(1, 1)):
  filters1, filters2, filters3 = filters
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = tf.keras.layers.Conv2D(filters1, (1, 1), strides=strides, kernel_initializer='he_normal',name=conv_name_base + '2a')(input_tensor)
  x = tf.keras.layers.BatchNormalization(name=bn_name_base + '2a')(x)
  x = tf.keras.layers.Activation('relu')(x)

  x = tf.keras.layers.Conv2D(filters2, (kernel_size,1), padding='same',kernel_initializer='he_normal', name=conv_name_base + '2b')(x)
  x = tf.keras.layers.BatchNormalization(name=bn_name_base + '2b')(x)
  x = tf.keras.layers.Activation('relu')(x)

  x = tf.keras.layers.Conv2D(filters3, (1, 1),kernel_initializer='he_normal',name=conv_name_base + '2c')(x)
  x = tf.keras.layers.BatchNormalization(name=bn_name_base + '2c')(x)

  shortcut = tf.keras.layers.Conv2D(filters3, (1, 1), strides=strides, kernel_initializer='he_normal',name=conv_name_base + '1')(input_tensor)
  shortcut = tf.keras.layers.BatchNormalization(name=bn_name_base + '1')(shortcut)

  x = tf.keras.layers.add([x, shortcut])
  x = tf.keras.layers.Activation('relu')(x)
  return x

def ResNet50():
  img_input = tf.keras.layers.Input(shape=(187,2,1))
  x = tf.keras.layers.Conv2D(128, (10, 2),strides=(5, 1),padding='valid',kernel_initializer='he_normal',name='conv1')(img_input)
  x = tf.keras.layers.BatchNormalization(name='bn_conv1')(x)
  x = tf.keras.layers.Activation('relu')(x)

  x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
  x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
  x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

  x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

  x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

  x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
  x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
  x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

  x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
  x = tf.keras.layers.Dropout(rate=0.9)(x)
  x = tf.keras.layers.Dense(5, activation='softmax', name='fc5')(x)

  model = tf.keras.models.Model(img_input, x, name='resnet50')
  model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
  print(model.summary())
  return model


def return_v05_model():
    input_tens = tf.keras.Input(shape=(187,2,1))
    x = tf.keras.layers.Conv2D(512, kernel_size=(10,2), strides=(5,1),padding='valid')(input_tens)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Conv2D(1024, kernel_size=(5,1), padding='valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Conv2D(1024, kernel_size=(5,1), padding='valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=(5,1), padding='valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=(5,1), padding='valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(5,1), padding='valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(5, activation="softmax")(x)
    model = tf.keras.Model(inputs=input_tens, outputs=x)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
    print(model.summary())
    return model

def return_v06_model():
    input_tens = tf.keras.Input(shape=(187,2,1))
    x = tf.keras.layers.Conv2D(128, kernel_size=(5,2), strides=(2,1), padding='valid')(input_tens)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=(5,1), strides=(2,1), padding='valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=(3,1), padding='valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=(3,1), padding='valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=(3,1), padding='valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=(3,1), padding='valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Conv2D(1024, kernel_size=(3,1), padding='valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Conv2D(1024, kernel_size=(3,1), padding='valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Dense(5, activation="softmax")(x)
    model = tf.keras.Model(inputs=input_tens, outputs=x)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
    print(model.summary())
    return model