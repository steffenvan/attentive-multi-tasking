import tensorflow as tf


def split_heads_2d(inputs, num_heads):
    s = inputs.shape[:-1]
    ret_shape = [-1, s[1], s[2], num_heads, inputs.shape[-1] // num_heads]
    split = tf.reshape(inputs, ret_shape)
    return tf.transpose(split, [0, 3, 1, 2, 4])

def combine_heads_2d(inputs):
    # inputs: [batch, num_heads, H, W, dim_keys]
    transposed = tf.transpose(inputs, [0, 2, 3, 1, 4])
    # [batch_, H, W, num_heads, dim_keys]
    num_heads, dim_keys = transposed.shape[-2:]
    ret_shape = [-1, transposed.shape[1], transposed.shape[2], num_heads * dim_keys]
    return tf.reshape(transposed, ret_shape)

def compute_flat_qkv(inputs, dim_keys, dim_values, num_heads):
    _, H, W, _ = inputs.shape
    qkv = tf.layers.conv2d(inputs, 2*dim_keys + dim_values, 1)
    q, k, v = tf.split(qkv, [dim_keys, dim_keys, dim_values], axis=3)
    q = split_heads_2d(q, num_heads)
    k = split_heads_2d(k, num_heads)
    v = split_heads_2d(v, num_heads)
    dim_keys_heads = dim_keys // num_heads
    q *= dim_keys_heads ** -0.5
    flat_q = tf.reshape(q, [-1, num_heads, H * W, dim_keys/num_heads])
    flat_k = tf.reshape(k, [-1, num_heads, H * W, dim_keys/num_heads])
    flat_v = tf.reshape(v, [-1, num_heads, H * W, dim_values/num_heads])
    return flat_q, flat_k, flat_v, H, W

def relative_logits_1d(q, rel_k, H, W, num_heads, transpose_mask):
    rel_logits = tf.einsum('bhsd,md->bhsm', q, rel_k)
    rel_logits = tf.reshape(rel_logits, [-1, num_heads*H, W, 2*W-1])
    rel_logits = rel_to_abs(rel_logits)
    rel_logits = tf.reshape(rel_logits, [-1, num_heads, H, W, W])
    rel_logits = tf.expand_dims(rel_logits, axis=3)
    rel_logits = tf.tile(rel_logits, [1, 1, 1, H, 1, 1])
    rel_logits = tf.transpose(rel_logits, transpose_mask)
    rel_logits = tf.reshape(rel_logits, [-1, num_heads, H*W, H*W])
    return rel_logits

def relative_logits(q, H, W):
    _, num_heads, _ , dim_keys = q.shape

    key_rel_w = tf.get_variable(
        'key_rel_w', shape=(2*W-1, dim_keys),
        initializer=tf.random_normal_initializer(tf.pow(tf.to_float(dim_keys), -0.5))
    )
    rel_logits_w = relative_logits_1d(
        q, key_rel_w, H, W, num_heads, [0, 1, 2, 4, 3, 5]
    )

    # Relative logits in height dimension.
    # For ease, we transpose height and width and repeat the
    # above steps, and transpose to eventually put the logits
    # in the correct positions.
    key_rel_h = tf.get_variable(
        'key_rel_h', shape=(2*H-1, dim_keys),
        initializer=tf.random_normal_initializer(tf.pow(tf.to_float(dim_keys), -0.5))
    )

    q = tf.reshape(q, [-1, num_heads, H, W, dim_keys])
    q = tf.transpose(q, [0, 1, 3, 2, 4])
    q = tf.reshape(q, [-1, num_heads, H*W, dim_keys])
    rel_logits_h = relative_logits_1d(
        q,
        key_rel_h, W, H, num_heads, [0, 1, 4, 2, 5, 3]
    )
    return rel_logits_h, rel_logits_w


def rel_to_abs(x):
    _, num_heads, L, _ = x.shape
    x = tf.pad(x, paddings=[[0, 0], [0,0], [0, 0], [0, 1]])
    flat_x = tf.reshape(x, [-1, num_heads, L * 2 * L])
    flat_x_padded = tf.pad(flat_x, paddings=[[0,0], [0, 0], [0, L-1]])
    final_x = tf.reshape(flat_x_padded, [-1, num_heads, L+1, 2*L-1])
    final_x = final_x[:, :, :L, L-1:]
    return final_x


def augmented_conv2d(inputs, output_channels, kernel_shape, dim_keys, dim_values, num_heads, use_relative, batch_size):
    
    conv_out = tf.layers.conv2d(inputs, output_channels - dim_values, kernel_shape)
    flat_q, flat_k, flat_v, H, W = compute_flat_qkv(inputs, dim_keys, dim_values, num_heads)
    logits = tf.matmul(flat_q, flat_k, transpose_b=True)
    if use_relative:
        h_rel_logits, w_rel_logits = relative_logits(flat_q, H, W) 
        logits += h_rel_logits
        logits += w_rel_logits
    weights = tf.nn.softmax(logits)
    attn_out = tf.matmul(weights, flat_v)
    attn_out = tf.reshape(attn_out, [batch_size, num_heads, H, W, dim_values // num_heads])
    attn_out = combine_heads_2d(attn_out)
    attn_out = tf.layers.conv2d(attn_out, dim_values, 1)
    conv_out = tf.image.resize_bilinear(conv_out, size=inputs.shape[1:3])
    return tf.concat([conv_out, attn_out], axis=3)