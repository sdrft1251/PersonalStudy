import tensorflow as tf

def convdu(input_tens, channel, reduction):
    x = tf.keras.layers.Conv2D(channel//reduction, kernel_size=(1,1), padding="valid", activation="relu")(input_tens)
    x = tf.keras.layers.Conv2D(channel//reduction, kernel_size=(1,1), padding="valid", activation="sigmoid")(x)
    return x


def covpool(input_tens):
    x = input_tens
    batchSize = input_tens.shape[0]
    h = input_tens.shape[1]
    w = input_tens.shape[2]
    c = input_tens.shape[3]
    M = h*w
    x = x.reshape(batchSize, c, M)
    I_hat = (-1./M/M)*tf.ones(M,M) + (1./M)*tf.eye(M,M)
    I_hat = tf.repeat(I_hat.reshape(1,M,M), repeats=batchSize, axis=0)
    y = tf.linalg.matmul(tf.linalg.matmul(x, I_hat), tf.transpose(x, perm=[2,1]))
    return y

def sqrtm(input_tens, iterN):
    x = input_tens
    batchSize = input_tens.shape[0]
    c = input_tens.shape[1]
    I3 = tf.repeat((3.0*tf.eye(c, c)).reshape(1,c,c), repeats=batchSize, axis=0)
    normA = tf.math.reduce_sum(tf.math.reduce_sum((1.0/3.0)*tf.math.multiply(x, I3), axis=1), axis=1)
    A = tf.math.divide(x, tf.broadcast_to(normA.reshape(batchSize, 1, 1), shape=x.shape))
    Y = tf.zeros(batchSize, iterN, c, c)
    Z = tf.repeat(tf.eye(c,c).reshape(1,c,c), repeats=[batchSize, iterN], axis=0)
    if iterN < 2:
        ZY = 0.5*(I3 - A)
        Y[:,0,:,:] = tf.linalg.matmul(A, ZY)
    else:
        ZY = 0.5*(I3 - A)
        Y[:,0,:,:] = tf.linalg.matmul(A, ZY)
        Z[:,0,:,:] = ZY
        for i in range(1, iterN-1):
            ZY = 0.5*(I3 - tf.linalg.matmul(Z[:,i-1,:,:], Y[:,i-1,:,:]))
            Y[:,i,:,:] = tf.linalg.matmul(Y[:,i-1,:,:], ZY)
            Z[:,i,:,:] = tf.linalg.matmul(ZY, Z[:,i-1,:,:])
        ZY = tf.linalg.matmul(0.5*Y[:,iterN-2,:,:], (I3-tf.linalg.matmul(Z[:,iterN-2,:,:],Y[:,iterN-2,:,:])))
    y = ZY*(tf.broadcast_to(tf.math.sqrt(normA).reshape(batchSize,1,1), shape=x.shape))
    return y


def _embedded_gaussian(input_tens):
    batchSize, h, w, c = input_tens.shape
    g_x = g(input_tens).reshape(batchSize, inter_channels, -1)


def g(input_tens, out_channels, sub_sample):
    g_x = tf.keras.layers.Conv2D(out_channels, kernel_size=(1,1), padding="valid")(input_tens)
    if sub_sample:
        g_x = tf.keras.layers.MaxPool2D(pool_size=(2,2))(g_x)
    return g_x