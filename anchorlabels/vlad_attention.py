from tensorflow import flags
import tensorflow as tf
import frame_level_models
import video_level_models
import models
import model_utils as utils
import tensorflow.contrib.slim as slim
from tensorflow import  matmul, reshape, shape, transpose, cast, float32,logging
from tensorflow.layers import Dense, Layer
import math
import numpy as np
from warnings import filterwarnings
filterwarnings("ignore")
FLAGS = flags.FLAGS

flags.DEFINE_bool(
    "add_batch_norm_attn", False,
    "batch normalisation needed for attention")
flags.DEFINE_bool(
            "early_attention", False,
            "Turning on early attention")
flags.DEFINE_bool(
            "shift_operation", False,
            "turning on shift opertaion for attention ")
flags.DEFINE_bool(
            "late_attention", False,
            "turning on late attention")
flags.DEFINE_float(
            "dropout_rate",0,
            "drop out rate for attention")
flags.DEFINE_integer(
            "numofheads",1,
            "number of heads for attention")
flags.DEFINE_bool(
                    "very_early_attention", False,
                    "Turning on very early attention")


# Implementing the Scaled-Dot Product Attention
class DotProductAttention(Layer):
    def __init__(self, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
 
    def call(self, queries, keys, values, d_k, mask=None):
        dim=  cast(d_k, tf.float32)
        # Scoring the queries against the keys after transposing the latter, and scaling
        scores = matmul(queries, keys, transpose_b=True) / tf.math.sqrt(dim)
       
        # Apply mask to the attention scores
        if mask is not None:
            scores += -1e9 * mask

        # Computing the weights by a softmax operation
        weights = tf.nn.softmax(scores)
 
        # Computing the attention by a weighted sum of the value vectors
        return matmul(weights, values)

# Implementing the Multi-Head Attention
class MultiHeadAttention(Layer):
    def __init__(self, h, d_k, d_v, d_model,shift_size ,is_training,**kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.attention = DotProductAttention()  # Scaled dot product attention
        self.heads = h  # Number of attention heads to use
        self.d_k = d_k  # Dimensionality of the linearly projected queries and keys
        self.d_v = d_v  # Dimensionality of the linearly projected values
        self.d_model = d_model  # Dimensionality of the model
        self.depth = self.d_model // self.heads
        self.shift_size=shift_size
        self.early_attention=FLAGS.early_attention
        self.shift_operation =FLAGS.shift_operation
        self.add_batch_norm = FLAGS.add_batch_norm_attn
        self.dropout_rate = FLAGS.dropout_rate
        self.is_training = is_training

        self.W_q = Dense(d_k,use_bias=False,kernel_constraint=None,activation = None)  # Learned projection matrix for the queries
        self.W_k = Dense(d_k,use_bias=False,kernel_constraint=None,activation = None)  # Learned projection matrix for the keys
        self.W_v = Dense(d_v,use_bias=False,kernel_constraint=None,activation = None)  # Learned projection matrix for the values
        self.W_o = Dense(d_model,use_bias=False,kernel_constraint=None,activation = None)  # Learned projection matrix for the multi-head output


    def reshape_tensor(self, x, heads, flag):
        if flag:
            # Tensor shape after reshaping and transposing: (batch_size, heads, seq_length, -1)
            x = reshape(x, shape=(shape(x)[0], -1, heads, self.depth))
            #x = reshape(x, shape=(shape(x)[0], shape(x)[1], heads, -1))
            x = transpose(x, perm=(0, 2, 1, 3))
        else:
            # Reverting the reshaping and transposing operations: (batch_size, seq_length, d_k)
            x = transpose(x, perm=(0, 2, 1, 3))
            #x = reshape(x, shape=(shape(x)[0], shape(x)[1],self.d_model))
            x = reshape(x, shape=(shape(x)[0], -1,self.d_model))
        if self.add_batch_norm:    
            x = tf.layers.batch_normalization(x, training=self.is_training)
        if self.is_training:
            x = tf.layers.dropout(x, self.dropout_rate)
        return x
 
    def call(self, queries,keys,values, mask=None):
        # Rearrange the queries to be able to compute all heads in parallel
        q_reshaped = self.reshape_tensor(self.W_q(queries), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
 
        # Rearrange the keys to be able to compute all heads in parallel
        k_reshaped = self.reshape_tensor(self.W_k(keys), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
 
        # Rearrange the values to be able to compute all heads in parallel
        v_reshaped = self.reshape_tensor(self.W_v(values), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
 
        # Compute the multi-head attention output using the reshaped queries, keys and values
        o_reshaped = self.attention(q_reshaped, k_reshaped, v_reshaped, self.d_k, mask)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
 
        # Rearrange back the output into concatenated form
        output = self.reshape_tensor(o_reshaped, self.heads, False)
        # Resulting tensor shape: (batch_size, input_seq_length, d_v)
        
        if self.shift_operation:
            normalized_output = tf.nn.l2_normalize(output, 1)
            alpha = tf.get_variable("alpha",
                                    [self.shift_size],
                                    initializer=tf.constant_initializer(1.0))
            beta = tf.get_variable("beta",
                                   [self.shift_size],
                                   initializer=tf.constant_initializer(0.0))
            output = tf.multiply(normalized_output, alpha)
            output = tf.add(output, beta)

        # Apply one final linear projection to the output to generate the multi-head attention
        # Resulting tensor shape: (batch_size, input_seq_length, d_model)

        normalized_output = tf.nn.l2_normalize(output, 1)
        normalized_output = tf.reshape(normalized_output, [-1, self.d_model])
        output = tf.nn.l2_normalize(normalized_output)

        return self.W_o(output)

class NetVLADAttnModel(models.BaseModel):
  """Creates a NetVLAD based model with attention.

  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).

  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """
  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.netvlad_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.netvlad_cluster_size
    hidden1_size = hidden_size or FLAGS.netvlad_hidden_size
    relu = FLAGS.netvlad_relu
    dimred = FLAGS.netvlad_dimred
    gating = FLAGS.gating
    remove_diag = FLAGS.gating_remove_diag
    very_early_attention=FLAGS.very_early_attention
    early_attention=FLAGS.early_attention
    late_attention=FLAGS.late_attention
    heads=FLAGS.numofheads

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)


    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])
    if very_early_attention:
         video_input= reshaped_input[:, 0:1024]
         video_features = tf.nn.l2_normalize(video_input, 1)
         video_features = tf.reshape(video_features, [-1, max_frames, 1024])
         video_attn_layer =  MultiHeadAttention(heads, d_k=video_features.get_shape()[2],d_v=video_features.get_shape()[2], d_model=video_features.get_shape()[2],shift_size=1024,is_training=is_training)
         video_attn=video_attn_layer(video_features,video_features,video_features)
         video_input=tf.reshape(video_attn,[-1,1024])
   
         audio_input=reshaped_input[:, 1024:]
         audio_features = tf.nn.l2_normalize(audio_input, 1)
         audio_features = tf.reshape(audio_features, [-1, max_frames, 128])
         audio_attn_layer =  MultiHeadAttention(heads, d_k=audio_features.get_shape()[2], d_v=audio_features.get_shape()[2],d_model=audio_features.get_shape()[2],shift_size=128,is_training=is_training)
         audio_attn=audio_attn_layer(audio_features,audio_features,audio_features)
         audio_input=tf.reshape(audio_attn,[-1,128]) 

    video_NetVLAD = frame_level_models.NetVLAD(1024,max_frames,cluster_size, add_batch_norm, is_training)
    audio_NetVLAD = frame_level_models.NetVLAD(128,max_frames,cluster_size/2, add_batch_norm, is_training)

    if (add_batch_norm and not very_early_attention):# and not lightvlad:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")
    if (not very_early_attention):
        video_input= reshaped_input[:, 0:1024]
        audio_input=reshaped_input[:, 1024:]

    with tf.variable_scope("video_VLAD"):
        vlad_video = video_NetVLAD.forward(video_input)

    with tf.variable_scope("audio_VLAD"):
        vlad_audio = audio_NetVLAD.forward(audio_input)

    if early_attention:
         video_features = tf.reshape(vlad_video,[-1 ,cluster_size ])
         video_attn_layer =  MultiHeadAttention(heads, d_k=video_features.get_shape()[1],d_v=video_features.get_shape()[1], d_model=video_features.get_shape()[1],shift_size=cluster_size,is_training=is_training)
         vlad_video_attn=video_attn_layer(video_features,video_features,video_features)
         vlad_video=tf.reshape(vlad_video_attn,[-1,cluster_size*1024])

         audio_features = tf.reshape(vlad_audio,[-1,cluster_size//2])
         audio_attn_layer =  MultiHeadAttention(heads, d_k=audio_features.get_shape()[1], d_v=audio_features.get_shape()[1],d_model=audio_features.get_shape()[1],shift_size=cluster_size//2,is_training=is_training)
         vlad_audio_attn=audio_attn_layer(audio_features,audio_features,audio_features)
         vlad_audio=tf.reshape(vlad_audio_attn,[-1,(cluster_size//2) * 128])


    vlad = tf.concat([vlad_video, vlad_audio],1)
    vlad_dim = vlad.get_shape().as_list()[1]

    cs=cluster_size
    hidden1_weights = tf.get_variable("hidden1_weights", [vlad_dim, hidden1_size],initializer=tf.random_normal_initializer(stddev=1/math.sqrt(cs)))

    activation = tf.matmul(vlad, hidden1_weights)

    if add_batch_norm and relu:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")

    else:
      hidden1_biases = tf.get_variable("hidden1_biases",
        [hidden1_size],
        initializer = tf.random_normal_initializer(stddev=0.01))
      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases

    if relu:
      activation = tf.nn.relu6(activation)
    
    if(late_attention):
         vlad_features = activation 
         vlad_feature_attn_layer =  MultiHeadAttention(heads, d_k=vlad_features.get_shape()[1],d_v=vlad_features.get_shape()[1], d_model=vlad_features.get_shape()[1],shift_size=1024,is_training=is_training)
         vlad_feature_attn=vlad_feature_attn_layer(vlad_features,vlad_features,vlad_features)
         activation=tf.reshape(vlad_feature_attn,[-1,activation.get_shape()[1]])

    if (gating and not (early_attention or late_attention or very_early_attention)):
        gating_weights = tf.get_variable("gating_weights_2",
          [hidden1_size, hidden1_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(hidden1_size)))

        gates = tf.matmul(activation, gating_weights)

        if remove_diag:
            #removes diagonals coefficients
            diagonals = tf.matrix_diag_part(gating_weights)
            gates = gates - tf.multiply(diagonals,activation)


        if add_batch_norm:
          gates = slim.batch_norm(
              gates,
              center=True,
              scale=True,
              is_training=is_training,
              scope="gating_bn")
        else:
          gating_biases = tf.get_variable("gating_biases",
            [cluster_size],
            initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
          gates += gating_biases

        gates = tf.sigmoid(gates)

        activation = tf.multiply(activation,gates)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)


    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=is_training,
        **unused_params)  
