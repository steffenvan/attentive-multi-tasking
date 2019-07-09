import tensorflow as tf


def split_heads_2d(inputs, N_h):
    s = inputs.shape[:-1]
    print("S IS: ", s)
    print("Nh is: ", N_h)
    # s = tf.reshape(s, [3])
    ret_shape = [-1, s[1], s[2], N_h, inputs.shape[-1] / N_h]
    # ret_shape = tf.concat(values=[s, tf.TensorShape([N_h, tf.div(inputs.shape[-1], N_h)])], axis=0)
    split = tf.reshape(inputs, ret_shape)
    return tf.transpose(split, [0, 3, 1, 2, 4])


def combine_heads_2d(inputs):
    # inputs: [batch, N_h, H, W, d_k]
    transposed = tf.transpose(inputs, [0, 2, 3, 1, 4])
    # [batch_, H, W, N_h, d_k]
    print("TRANSPOSED: ", transposed)
    N_h, d_k = transposed.shape[-2:]
    # N_h, d_k
    # ret_shape = tf.concat(values=[transposed.shape[:-2], tf.reshape(N_h * d_k, [1])], axis=0)
    ret_shape = [-1, transposed.shape[1], transposed.shape[2], N_h * d_k]
    return tf.reshape(transposed, ret_shape)

def compute_flat_qkv(inputs, d_k, d_v, N_h):
    N, H, W, _ = inputs.shape
    qkv = tf.layers.conv2d(inputs, 2*d_k + d_v, 1)
    q, k, v = tf.split(qkv, [d_k, d_k, d_v], axis=3)
    q = split_heads_2d(q, N_h)
    k = split_heads_2d(k, N_h)
    v = split_heads_2d(v, N_h)
    d_kh = d_k // N_h
    q *= d_kh ** -0.5
    flat_q = tf.reshape(q, [-1, N_h, H * W, d_k/N_h])
    flat_k = tf.reshape(k, [-1, N_h, H * W, d_k/N_h])
    flat_v = tf.reshape(v, [-1, N_h, H * W, d_v/N_h])
    return flat_q, flat_k, flat_v, N, H, W

def relative_logits_1d(q, rel_k, H, W, N_h, transpose_mask):
    rel_logits = tf.einsum('bhsd,md->bhsm', q, rel_k)
    rel_logits = tf.reshape(rel_logits, [-1, N_h*H, W, 2*W-1])
    rel_logits = rel_to_abs(rel_logits)
    rel_logits = tf.reshape(rel_logits, [-1, N_h, H, W, W])
    rel_logits = tf.expand_dims(rel_logits, axis=3)
    rel_logits = tf.tile(rel_logits, [1, 1, 1, H, 1, 1])
    rel_logits = tf.transpose(rel_logits, transpose_mask)
    rel_logits = tf.reshape(rel_logits, [-1, N_h, H*W, H*W])
    return rel_logits

def relative_logits(q, H, W):
    print("Q IS: ", q)
    _, N_h, _ , d_k = q.shape

    key_rel_w = tf.get_variable(
        'key_rel_w', shape=(2*W-1, d_k),
        initializer=tf.random_normal_initializer(tf.pow(tf.to_float(d_k), -0.5))
    )
    rel_logits_w = relative_logits_1d(
        q, key_rel_w, H, W, N_h, [0, 1, 2, 4, 3, 5]
    )

    # Relative logits in height dimension.
    # For ease, we transpose height and width and repeat the
    # above steps, and transpose to eventually put the logits
    # in the correct positions.
    key_rel_h = tf.get_variable(
        'key_rel_h', shape=(2*H-1, d_k),
        initializer=tf.random_normal_initializer(tf.pow(tf.to_float(d_k), -0.5))
    )

    q = tf.reshape(q, [-1, N_h, H, W, d_k])
    q = tf.transpose(q, [0, 1, 3, 2, 4])
    q = tf.reshape(q, [-1, N_h, H*W, d_k])
    rel_logits_h = relative_logits_1d(
        q,
        key_rel_h, W, H, N_h, [0, 1, 4, 2, 5, 3]
    )
    return rel_logits_h, rel_logits_w


def rel_to_abs(x):
    B, N_h, L, _ = x.shape
    # col_pad = tf.zeros((tf.shape(x)[0], N_h, L, 1))

    # col_pad = tf.tile(col_pad, [B, 1, 1, 1])
    # x = tf.concat([x, col_pad], axis=3)
    x = tf.pad(x, paddings=[[0, 0], [0,0], [0, 0], [0, 1]])
    flat_x = tf.reshape(x, [-1, N_h, L * 2 * L])
    # flat_pad = tf.zeros((B, N_h, L-1))
    flat_x_padded = tf.pad(flat_x, paddings=[[0,0], [0, 0], [0, L-1]])
    # flat_x_padded = tf.concat([flat_x, flat_pad], axis=2)
    final_x = tf.reshape(flat_x_padded, [-1, N_h, L+1, 2*L-1])
    final_x = final_x[:, :, :L, L-1:]
    return final_x


def augmented_conv2d(X, Fout, k, d_k, d_v, N_h, relative, B):
    """
    Args:
         X: input image
         Fout: Output filters
         k: kernel_size
         d_k: dimension of keys
         d_v: dimension of values
         N_h: Number of attention heads
         relative: Whether to use relative position,
         B: Batch size
    """
    conv_out = tf.layers.conv2d(X, Fout - d_v, k)
    flat_q, flat_k, flat_v, N, H, W = compute_flat_qkv(X, d_k, d_v, N_h)
    logits = tf.matmul(flat_q, flat_k, transpose_b=True)
    if relative:
        h_rel_logits, w_rel_logits = relative_logits(flat_q, H, W) # error here: flat_q
        logits += h_rel_logits
        logits += w_rel_logits
    weights = tf.nn.softmax(logits)
    attn_out = tf.matmul(weights, flat_v)
    attn_out = tf.reshape(attn_out, [B, N_h, H, W, d_v // N_h]) # error here: flat_v
    attn_out = combine_heads_2d(attn_out)
    attn_out = tf.layers.conv2d(attn_out, d_v, 1)
    conv_out = tf.image.resize_bilinear(conv_out, size=X.shape[1:3])
    return tf.concat([conv_out, attn_out], axis=3)