import tensorflow as tf

def resBlock(x,channels=64,kernel=3,scale=1):
	tmp = tf.keras.layers.Conv2D(channels,kernel_size=(kernel,kernel), padding="same", activation="relu")(x)
	tmp = tf.keras.layers.Conv2D(channels,kernel_size=(kernel,kernel), padding="same", activation=None)(tmp)
	tmp *= scale
	return x + tmp

def upsample(x,scale=2,features=64, activation_method="relu"):
	assert scale in [2,3,4]
	x = tf.keras.layers.Conv2D(features, kernel_size=(3,3), padding="same", activation=activation_method)(x)
	if scale == 2:
		ps_features = 3*(scale**2)
		x = tf.keras.layers.Conv2D(ps_features,kernel_size=(3,3), padding="same",activation=activation_method)(x)
		x = PS(x,2,color=True)
	elif scale == 3:
		ps_features =3*(scale**2)
		x = tf.keras.layers.Conv2D(ps_features,kernel_size=(3,3), padding="same",activation=activation_method)(x)
		x = PS(x,3,color=True)
	elif scale == 4:
		ps_features = 3*(2**2)
		for i in range(2):
			x = tf.keras.layers.Conv2D(ps_features,kernel_size=(3,3), padding="same",activation=activation_method)(x)
			x = PS(x,2,color=True)
	return x

def _phase_shift(I, r):
	bsize, a, b, c = I.get_shape().as_list()
	bsize = tf.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
	X = tf.reshape(I, (bsize, a, b, r, r))
	X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
	X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
	X = tf.concat([tf.squeeze(x, axis=1) for x in X],2)  # bsize, b, a*r, r
	X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
	X = tf.concat([tf.squeeze(x, axis=1) for x in X],2)  # bsize, a*r, b*r
	return tf.reshape(X, (bsize, a*r, b*r, 1))

def PS(X, r, color=False):
	if color:
		Xc = tf.split(X, 3, 3)
		X = tf.concat([_phase_shift(x, r) for x in Xc],3)
	else:
		X = _phase_shift(X, r)
	return X

def log10(x):
  numerator = tf.math.log(x)
  denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator