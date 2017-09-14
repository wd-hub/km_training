import tensorflow as tf

def distance_matrix_vector(anchor, positive):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""

    d1_sq = tf.reduce_sum(tf.multiply(anchor, anchor), 1, keep_dims=True)
    d2_sq = tf.reduce_sum(tf.multiply(positive, positive), 1, keep_dims=True)

    eps = 1e-6
    return tf.sqrt((tf.tile(d1_sq, [1, anchor.get_shape().as_list()[0]])
                    + tf.transpose(tf.tile(d2_sq, [1, positive.get_shape().as_list()[0]]))
                    - 2.0*tf.matmul(anchor, tf.transpose(positive))) + eps)

def product_matrix_vector(anchor, positive):
    """Calculate product matrix"""
    return tf.matmul(anchor, tf.transpose(positive))

def loss_margin_min(anchor, positive, anchor_swap = False, anchor_ave = False, margin = 1.0):
    """HardNet margin loss - calculates loss based on distance matrix based on positive distance and closest negative distance.
    """
    assert anchor.get_shape().as_list() == positive.get_shape().as_list(), "Input sizes between positive and negative must be equal."
    assert anchor.get_shape().ndims == 2, "Input must be a 2D matrix."
    # params = dict()
    dist_matrix = distance_matrix_vector(anchor, positive)
    # params['distMat'] = dist_matrix
    eye = tf.eye(anchor.get_shape().as_list()[0])
    pos = tf.diag_part(dist_matrix)
    dist_without_min_on_diag = tf.add(dist_matrix, tf.multiply(eye, 10))
    # mask = (tf.cast(tf.greater(dist_without_min_on_diag, 0.008), tf.int32) - 1)*-1
    # mask = tf.multiply(tf.cast(mask, dist_without_min_on_diag.dtype), 10)
    # dist_without_min_on_diag = dist_without_min_on_diag + mask
    min_neg = tf.reduce_min(dist_without_min_on_diag, 1)
    min_idx = tf.arg_min(dist_without_min_on_diag, 1)
    if anchor_swap:
        min_neg2 = tf.reduce_min(dist_without_min_on_diag, 0)
        min_idx = tf.arg_min(dist_without_min_on_diag, 0)
        min_neg = tf.minimum(min_neg, min_neg2)
    dist_hinge = tf.maximum(0., margin + pos - min_neg)
    # from vincent cvpr15
    # dist_hinge = tf.maximum(0., 1 - min_neg/(pos+margin))
    # dist_hinge = tf.maximum(0., 1 - min_neg/(pos+margin)) + pos

    if anchor_ave:
        min_neg2 = tf.minimum(dist_without_min_on_diag, 0)
        dist_hinge2 = tf.maximum(0., 1.0 + pos - min_neg2)
        dist_hinge = 0.5 * (dist_hinge2 + dist_hinge)
    mean_pos, var_pos = tf.nn.moments(pos, axes=[0])
    mean_neg, var_neg = tf.nn.moments(min_neg, axes=[0])

    watchParam = dict()
    # watchParam['mean_pos'] = mean_pos  # = tf.reduce_mean(pos)
    # watchParam['mean_neg'] = mean_neg
    watchParam['var_pos'] = var_pos
    watchParam['var_neg'] = var_neg
    watchParam['pos'] = tf.reduce_mean(pos)
    watchParam['neg'] = tf.reduce_mean(min_neg)
    loss = tf.reduce_mean(dist_hinge) + (var_pos + var_neg)

    #------------------------------------------------------#
    prod_matrix = product_matrix_vector(anchor, positive)
    # diagonal values are 0, off diagonal values are all 1
    mask_off_diag = tf.ones(anchor.get_shape().as_list()) - tf.eye(anchor.get_shape().as_list()[0])
    np_inner_product_matrix =tf.multiply(prod_matrix, mask_off_diag)
    M1 = tf.pow(tf.reduce_mean(np_inner_product_matrix), 2)
    M2 = tf.reduce_mean(tf.multiply(np_inner_product_matrix, np_inner_product_matrix))
    loss_gor = M1 + tf.maximum(0., tf.subtract(M2, 1/anchor.get_shape().as_list()[1]))

    loss = loss# + loss_gor
    watchParam['loss_gor'] = loss_gor
    watchParam['M1'] = M1
    watchParam['M2'] = M2
    # watchParam['prod_matrix'] = prod_matrix
    # watchParam['mask_off_diag'] = mask_off_diag
    return loss, min_idx, watchParam

def loss_metric(m_ap, m_an, margin=1.0):
    # m_ap: matric result of anchor & positive
    # m_an: matric result of anchor & negative
    d_p_squared = tf.square(m_ap)
    d_n_squared = tf.square(m_an)
    # loss = tf.maximum(0., margin - d_n_squared)
    loss = tf.maximum(0., margin + d_p_squared - d_n_squared)   # 0:positve pair, 1:negative pair
    return tf.reduce_mean(loss), tf.reduce_mean(d_p_squared), tf.reduce_mean(d_n_squared)

def distance_matrix_cosine(anchor, positive):
    return tf.matmul(anchor, tf.transpose(positive))

def loss_margin_cosine(anchor, positive, anchor_swap = False, anchor_ave = False, margin = 1.0):
    assert anchor.get_shape().as_list() == positive.get_shape().as_list(), "Input sizes between positive and negative must be equal."
    assert anchor.get_shape().ndims == 2, "Inputd must be a 2D matrix."
    dist_matrix = distance_matrix_cosine(anchor, positive)
    params = dict()
    params['distMat'] = dist_matrix

    eye = tf.eye(anchor.get_shape().as_list()[0])
    pos = tf.diag_part(dist_matrix)
    dist_without_max_on_diag = tf.subtract(dist_matrix, tf.multiply(eye, 10))

    max_neg = tf.reduce_max(dist_without_max_on_diag, 1)
    dist_hinge = tf.maximum(0., 2 - pos + max_neg)
    loss = tf.reduce_mean(dist_hinge)
    params['loss'] = loss
    return params

############################### tfeat loss #######################################
def compute_euclidean_distance(x, y):
    """
    Computes the euclidean distance between two tensorflow variables
    """

    d = tf.square(tf.subtract(x, y))
    d = tf.sqrt(tf.reduce_sum(d, 1)) # What about the axis ???
    return d


def compute_triplet_loss(anchor_feature, positive_feature, negative_feature, margin):
    """
    Compute the contrastive loss as in

    L = || f_a - f_p ||^2 - || f_a - f_n ||^2 + m

    **Parameters**
     anchor_feature:
     positive_feature:
     negative_feature:
     margin: Triplet margin

    **Returns**
     Return the loss operation
    """

    with tf.name_scope("triplet_loss"):
        d_p_squared = tf.reduce_sum(tf.square(anchor_feature - positive_feature), 1)
        d_n_squared = tf.reduce_sum(tf.square(anchor_feature - negative_feature), 1)

        dist_hinge = tf.maximum(0., margin + d_p_squared - d_n_squared)
        mean_pos, var_pos = tf.nn.moments(d_p_squared, axes=[0])
        mean_neg, var_neg = tf.nn.moments(d_n_squared, axes=[0])
        loss = tf.reduce_mean(dist_hinge)+(var_pos+var_neg)

        return loss, tf.reduce_mean(d_p_squared), tf.reduce_mean(d_n_squared)

def compute_triplet_loss_swap(anchor_feature, positive_feature, negative_feature, margin):
    with tf.name_scope("triplet_loss_swap"):
        d_p_squared = tf.reduce_sum(tf.square(anchor_feature - positive_feature), 1)
        d_n_squared = tf.reduce_sum(tf.square(anchor_feature - negative_feature), 1)
        d_h_squared = tf.reduce_sum(tf.square(positive_feature - negative_feature), 1)

        d_star = tf.minimum(d_n_squared, d_h_squared)
        loss = tf.maximum(0., margin + d_p_squared - d_star)

        return tf.reduce_mean(loss), tf.reduce_mean(d_p_squared), tf.reduce_mean(d_n_squared)