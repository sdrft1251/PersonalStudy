import tensorflow as tf

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model
        # 정확히 분배되는지 확인
        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads
        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        self.dense = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value = inputs['query'], inputs['key'], inputs['value']
        batch_size = tf.shape(query)[0]

        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention, _ = scaled_dot_product_attention(query, key, value)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        outputs = self.dense(concat_attention)
        return outputs

def scaled_dot_product_attention(query, key, value):
    # Get Q * K
    matmul_qk = tf.linalg.matmul(query, key, transpose_b=True)
    # For Scaling
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)
    # Get Attention Weights
    attention_weights = tf.nn.softmax(logits, axis=-1)
    outputs = tf.linalg.matmul(attention_weights, value)
    return outputs, attention_weights

class InputAttention(Layer):
    def __init__(self, T):
        super(InputAttention, self).__init__(name="input_attention")
        self.w1 = Dense(T)
        self.w2 = Dense(T)
        self.v = Dense(1)

    def call(self, h_s, c_s, x):
        """
        h_s : hidden_state (shape = batch,hidden_dims)
        c_s : cell_state (shape = batch,hidden_dims)
        x : time series encoder inputs (shape = batch,T,1)
        """
        query = tf.concat([h_s, c_s], axis=-1)  # batch, hidden_dims*2
        query = RepeatVector(x.shape[2])(query)  # batch, 1, hidden_dims*2
        x_perm = Permute((2, 1))(x)  # batch, 1, T
        score = tf.nn.tanh(self.w1(x_perm) + self.w2(query))  # batch, 1, T
        score = tf.squeeze(score)  # batch, T
        attention_weights = tf.nn.softmax(score)  # t 번째 time step 일 때 각 time step 별 중요도
        return attention_weights

class TemporalAttention(Layer):
    def __init__(self, latent_dims):
        super(TemporalAttention, self).__init__(name="temporal_attention")
        self.w1 = Dense(latent_dims)
        self.w2 = Dense(latent_dims)
        self.v = Dense(1)

    def call(self, h_s, c_s, latent):
        """
        h_s : hidden_state (shape = batch, hidden_dims)
        c_s : cell_state (shape = batch, hidden_dims)
        latent : time series encoder inputs (shape = batch, latent_dims)
        """
        query = tf.concat([h_s, c_s], axis=-1)  # batch, hidden_dims*2
        score = tf.nn.tanh(self.w1(latent) + self.w2(query))  # batch, latent_dims
        attention_weights = tf.nn.softmax(
            score, axis=1
        )  # Latent Space 안에의 중요성 # batch, latent_dims
        return attention_weights