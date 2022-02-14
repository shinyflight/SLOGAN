import argparse
import datetime
from functools import partial
import os
import numpy as np
import math
import time
import pickle
from tqdm import tqdm
import tensorflow_probability as tfp
import tensorflow as tf
from sklearn import metrics
from sklearn.cluster import KMeans
from tensorflow.python.ops import array_ops
from intra_cluster_fid import calculate_cluster_distance
from augmentation import DiffAugment

tfd = tfp.distributions
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
map_fn = tf.map_fn

# DATA
class ImageDataSet(object):
    def __init__(self, images, labels):
        assert images.ndim == 4

        self.num_examples = images.shape[0]

        self.images = images
        self.labels = labels
        self.epochs_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size):
        assert batch_size <= self.num_examples

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.num_examples:
            # Finished epoch
            self.epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch
        return self.images[start:end], self.labels[start:end]


def unpickle(file):
    with open(file, 'rb') as fo:
        dct = pickle.load(fo, encoding='bytes')
    return dct


def get_cifar10_dataset(split=None):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    data_arr = np.concatenate([x_train, x_test], axis=0)
    raw_float = np.array(data_arr, dtype='float32') / 256.0
    images = raw_float.reshape([-1, 32, 32, 3])
    labels = np.concatenate([y_train, y_test], axis=0)[:,0]

    images = images[(labels==0) | (labels==6)]
    labels = labels[(labels==0) | (labels==6)]
    labels[labels==6] = 1

    perm = np.arange(len(images))
    np.random.shuffle(perm)
    images = images[perm]
    labels = labels[perm]

    if args.ratio_plane < 5:
        plane_idx = np.where(labels==0)
        drop_idx = np.random.choice(plane_idx[0], size=len(plane_idx[0]) - int(len(plane_idx[0]) * (args.ratio_plane / (10. - args.ratio_plane))), replace=False)
        images = np.delete(images, drop_idx, axis=0)
        labels = np.delete(labels, drop_idx, axis=0)
    elif args.ratio_plane > 5 :
        frog_idx = np.where(labels==1)
        drop_idx = np.random.choice(frog_idx[0], size=len(frog_idx[0]) - int(len(frog_idx[0]) * ((10. - args.ratio_plane) / args.ratio_plane)), replace=False)
        images = np.delete(images, drop_idx, axis=0)
        labels = np.delete(labels, drop_idx, axis=0)

    labels = np.eye(args.num_cluster)[labels]

    if split is None:
        pass
    elif split == 'train':
        images = images[:-2000]
        labels = labels[:-2000]
    elif split == 'test':
        images = images[-2000:]
        labels = labels[-2000:]
    else:
        raise ValueError('unknown split')

    dataset = ImageDataSet(images, labels)

    return dataset


def random_flip(x, up_down=False, left_right=True):
    with tf.name_scope('random_flip'):
        s = tf.shape(x)
        if up_down:
            mask = tf.random_uniform([s[0], 1, 1, 1], 0.0, 1.0)
            mask = tf.tile(mask, [1, s[1], s[2], s[3]])
            x = tf.where(mask < 0.5, x, tf.reverse(x, axis=[1]))
        if left_right:
            mask = tf.random_uniform([s[0], 1, 1, 1], 0.0, 1.0)
            mask = tf.tile(mask, [1, s[1], s[2], s[3]])
            x = tf.where(mask < 0.5, x, tf.reverse(x, axis=[2]))
        return x


def get_cifar10_tf(sess, batch_size=1, shape=[32, 32], split=None, augment=True, start_queue_runner=True):
    with tf.name_scope('get_cifar10_tf'):
        dataset = get_cifar10_dataset(split=split)
        image_placeholder = tf.placeholder(dataset.images.dtype, dataset.images.shape)
        label_placeholder = tf.placeholder(dataset.labels.dtype, dataset.labels.shape)
        data = tf.data.Dataset.from_tensor_slices((image_placeholder, label_placeholder))
        data = data.repeat(None)
        data = data.prefetch(tf.data.experimental.AUTOTUNE)
        data = data.shuffle(buffer_size=100000)
        data = data.batch(batch_size)
        iterator = data.make_initializable_iterator()
        data_batch = iterator.get_next()
        sess.run(iterator.initializer, feed_dict={image_placeholder: dataset.images, label_placeholder: dataset.labels})

        if augment:
            data_batch = list(data_batch)
            data_batch[0] = random_flip(data_batch[0])
            data_batch[0] += tf.random_uniform(tf.shape(data_batch[0]), 0.0, 1.0 / 256.0)
            data_batch = tuple(data_batch)

        if start_queue_runner:
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord)

        return data_batch

def create_sprite(data):
    """
    Tile images into sprite image.
    Add any necessary padding
    """

    # For B&W or greyscale images
    if len(data.shape) == 3:
        data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0), (0, 0), (0, 0))
    data = np.pad(data, padding, mode='constant',
                  constant_values=0)

    # Tile images into sprite
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3, 4))
    # print(data.shape) => (n, image_height, n, image_width, 3)

    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    # print(data.shape) => (n * image_height, n * image_width, 3)
    return data

def image_grid(x, size=8):
    t = tf.unstack(x[:size * size], num=size*size, axis=0)
    rows = [tf.concat(t[i*size:(i+1)*size], axis=0) for i in range(size)]
    image = tf.concat(rows, axis=1)
    return image[None]


def image_grid_summary(name, x):
    with tf.name_scope(name):
        tf.summary.image('grid', image_grid(x))


def scalars_summary(name, x):
    with tf.name_scope(name):
        x = tf.reshape(x, [-1])
        mean, var = tf.nn.moments(x, axes=0)
        tf.summary.scalar('mean', mean)
        tf.summary.scalar('std', tf.sqrt(var))


def apply_conv(x, filters=32, kernel_size=3, he_init=True, kernel_constraint=None):
    if he_init:
        initializer = tf.contrib.layers.variance_scaling_initializer(uniform=True)
    else:
        initializer = tf.contrib.layers.xavier_initializer(uniform=True)

    return tf.layers.conv2d(
        x, filters=filters, kernel_size=kernel_size, padding='SAME', kernel_initializer=initializer,
        kernel_constraint=kernel_constraint
    )


def activation(x):
    with tf.name_scope('activation'):
        return tf.nn.relu(x)


def bn(x):
    return tf.contrib.layers.batch_norm(
        x,
        decay=0.9,
        center=True,
        scale=True,
        epsilon=1e-5,
        zero_debias_moving_mean=True,
        is_training=is_training
    )


def stable_norm(x, ord):
    return tf.norm(tf.contrib.layers.flatten(x), ord=ord, axis=1, keepdims=True)

def normalize(x, ord):
    return x / tf.maximum(tf.expand_dims(tf.expand_dims(stable_norm(x, ord=ord), -1), -1), 1e-10)


def downsample(x):
    with tf.name_scope('downsample'):
        x = tf.identity(x)
        return tf.add_n(
            [x[:,::2,::2,:], x[:,1::2,::2,:], x[:,::2,1::2,:], x[:,1::2,1::2,:]]
        ) / 4


def upsample(x):
    with tf.name_scope('upsample'):
        x = tf.identity(x)
        x = tf.concat([x, x, x, x], axis=-1)
        return tf.nn.depth_to_space(x, 2)


def conv_meanpool(x, **kwargs):
    return downsample(apply_conv(x, **kwargs))


def meanpool_conv(x, **kwargs):
    return apply_conv(downsample(x), **kwargs)


def upsample_conv(x, **kwargs):
    return apply_conv(upsample(x), **kwargs)


def l2_normalize(x, eps=1e-12):
    return x / tf.linalg.norm(x + eps)


def resblock(x, filters, resample=None, normalize=False, kernel_constraint=None):
    if normalize:
        norm_fn = bn
    else:
        norm_fn = tf.identity

    if resample == 'down':
        conv_1 = partial(apply_conv, filters=filters, kernel_constraint=kernel_constraint)
        conv_2 = partial(conv_meanpool, filters=filters, kernel_constraint=kernel_constraint)
        conv_shortcut = partial(conv_meanpool, filters=filters, kernel_size=1, he_init=False, kernel_constraint=kernel_constraint)
    elif resample == 'up':
        conv_1 = partial(upsample_conv, filters=filters, kernel_constraint=kernel_constraint)
        conv_2 = partial(apply_conv, filters=filters, kernel_constraint=kernel_constraint)
        conv_shortcut = partial(upsample_conv, filters=filters, kernel_size=1, he_init=False, kernel_constraint=kernel_constraint)
    elif resample == None:
        conv_1 = partial(apply_conv, filters=filters, kernel_constraint=kernel_constraint)
        conv_2 = partial(apply_conv, filters=filters, kernel_constraint=kernel_constraint)
        conv_shortcut = tf.identity

    with tf.name_scope('resblock'):
        x = tf.identity(x)
        update = conv_1(activation(norm_fn(x)))
        update = conv_2(activation(norm_fn(update)))

        skip = conv_shortcut(x)
        return skip + update


def resblock_optimized(x, filters, kernel_constraint=None):
    with tf.name_scope('resblock'):
        x = tf.identity(x)
        update = apply_conv(x, filters=filters, kernel_constraint=kernel_constraint)
        update = conv_meanpool(activation(update), filters=filters, kernel_constraint=kernel_constraint)
        # update = conv_meanpool(activation(bn(update)), filters=filters)

        skip = meanpool_conv(x, filters=filters, kernel_size=1, he_init=False, kernel_constraint=kernel_constraint)
        return skip + update


def generator(z, reuse):
    with tf.variable_scope('generator', reuse=reuse):
        channels = 128
        with tf.name_scope('pre_process'):
            z = tf.layers.dense(z, 4 * 4 * channels)
            x = tf.reshape(z, [-1, 4, 4, channels])

        with tf.name_scope('x1'):
            x = resblock(x, filters=channels, resample='up', normalize=True) # 8
            x = resblock(x, filters=channels, resample='up', normalize=True) # 16
            x = resblock(x, filters=channels, resample='up', normalize=True) # 32

        with tf.name_scope('post_process'):
            x = activation(bn(x))
            result = apply_conv(x, filters=3, he_init=False)
            return tf.tanh(result)


def discriminator(x, reuse):
    with tf.variable_scope('discriminator', reuse=reuse):
        with tf.name_scope('pre_process'):
            x = resblock_optimized(x, filters=128)

        with tf.name_scope('x1'):
            x = resblock(x, filters=128, resample='down') # 8
            x = resblock(x, filters=128) # 16
            x = resblock(x, filters=128) # 32

        with tf.name_scope('post_process'):
            x = activation(x)
            x = tf.reduce_mean(x, axis=[1, 2])
            flat = tf.contrib.layers.flatten(x)
            flat = tf.layers.dense(flat, 1)
            return flat

def encoder(x, out_dim, reuse):
    with tf.variable_scope('encoder', reuse=reuse):
        with tf.name_scope('pre_process'):
            x = resblock_optimized(x, filters=128)

        with tf.name_scope('x1'):
            x = resblock(x, filters=128, resample='down')  # 8
            x = resblock(x, filters=128)  # 16
            x = resblock(x, filters=128)  # 32

        with tf.name_scope('post_process'):
            x = activation(x)
            x = tf.reduce_mean(x, axis=[1, 2])
            flat = tf.contrib.layers.flatten(x)
            flat = tf.layers.dense(flat, out_dim)
            return flat


def remove_diag(M):
    h, w = M.shape
    assert h==w, "h and w should be same"
    mask = tf.cast(tf.ones_like(M) - tf.eye(h.value), tf.bool)
    return tf.reshape(tf.boolean_mask(M, mask), [h.value, -1])

def cosine_similarity(u, v):
    norm_u = tf.nn.l2_normalize(u, dim=1)
    norm_v = tf.nn.l2_normalize(v, dim=1)
    dist = tf.reduce_sum(tf.multiply(norm_u, norm_v), 1)
    return dist

def cosine_similarity_matrix(u, v):
    # normalize each row
    norm_u = tf.nn.l2_normalize(u, dim=1)
    norm_v = tf.nn.l2_normalize(v, dim=1)
    # element wise product
    prod = tf.matmul(norm_u, norm_v, adjoint_b=True)
    return prod

def sample_gumbel(shape, eps=1e-20):
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax(logits, temperature, hard=False):
    gumbel_softmax_sample = logits + sample_gumbel(tf.shape(logits))
    y = tf.nn.softmax(gumbel_softmax_sample / temperature)
    if hard:
        k = tf.shape(logits)[-1]
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y

def arcface(cos, s, m):
    cos_m = tf.math.cos(m)
    sin_m = tf.math.sin(m)
    cos_t2 = tf.square(cos, name='cos_2')
    sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
    sin_t = tf.sqrt(tf.clip_by_value(sin_t2,1e-10,1e10), name='sin_t')
    cos_mt = s * tf.subtract(tf.multiply(cos, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')
    threshold = tf.math.cos(math.pi - m)
    cond_v = cos - threshold
    cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)
    keep_val = s * (cos - tf.math.sin(m) * m)
    cos_mt_temp = tf.where(cond, cos_mt, keep_val)
    return cos_mt_temp


def cosine_contrastive_loss(cat_z, mu, enc_x, temp, s, m):
    with tf.variable_scope('contrastive_loss', reuse=None) as scope:
        mu_z = cat_z @ mu

        sim_enc_mu = arcface(cosine_similarity(enc_x, mu_z), s, m * temp)
        sim_enc_mu_mat = arcface(cosine_similarity_matrix(enc_x, mu_z), s, 0.0)

        numerator = tf.exp(sim_enc_mu)
        denominator = tf.clip_by_value(tf.reduce_mean(tf.concat([tf.expand_dims(tf.exp(sim_enc_mu),-1), remove_diag(tf.exp(sim_enc_mu_mat))], 1), 1),1e-10,1e10)
        criterion = -tf.log(tf.clip_by_value(numerator/denominator,1e-10,1e10))
        return criterion

def simclr_loss(enc_x, enc_x_):
    with tf.variable_scope('simclr_loss', reuse=None) as scope:
        numerator = tf.exp(cosine_similarity(enc_x, enc_x_))
        denominator = tf.clip_by_value(tf.reduce_sum(remove_diag(tf.exp(cosine_similarity_matrix(enc_x, enc_x_))), 1),1e-10,1e10)
        criterion = -tf.log(tf.clip_by_value(numerator/denominator,1e-10,1e10))
        return criterion

## evaluation metrics for cluster assignment
def calc_metrics(enc_x, y_test, num_cluster):
    km = KMeans(n_clusters=max(num_cluster, len(np.unique(y_test.argmax(axis=-1)))), random_state=0).fit(enc_x)
    labels_pred = km.labels_

    ari = metrics.adjusted_rand_score(y_test.argmax(axis=-1), labels_pred)
    nmi = metrics.normalized_mutual_info_score(y_test.argmax(axis=-1), labels_pred)
    return ari, nmi


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir",                   default="./logs/cifar2/")
    parser.add_argument("--gpu",           type=str,   default='0')
    parser.add_argument("--ratio_plane",   type=float, default=3)
    parser.add_argument("--coeff",         type=float, default=1)
    parser.add_argument("--lr_stein",      type=float, default=2e-3)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--scale",         type=float, default=4.0)
    parser.add_argument("--margin",        type=float, default=0.5)
    parser.add_argument("--start_temp",    type=float, default=1.0)
    parser.add_argument("--end_temp",      type=float, default=0.0)
    parser.add_argument("--pretrained",                 default=None)#'./logs/cifar2/3/pretrained/model-100000')
    parser.add_argument("--num_cluster",   type=int,   default=2)
    parser.add_argument("--log_freq",      type=int,   default=1000)
    parser.add_argument("--iterations",    type=int,   default=100000)
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--save_freq",     type=int,   default=100000)
    parser.add_argument("--val_freq",      type=int,   default=-1)
    parser.add_argument("--val_size",      type=int,   default=10000)
    parser.add_argument("--random_seed",   type=int,   default=0)
    parser.add_argument("--latent_dim",    type=int,   default=128)
    parser.add_argument("--lambda_lp",     type=float, default=10)
    parser.add_argument("--eps_min",       type=float, default=0.1)
    parser.add_argument("--eps_max",       type=float, default=10.0)
    parser.add_argument("--xi",            type=float, default=10.0)
    parser.add_argument("--ip",            type=int,   default=1)
    parser.add_argument("--K",             type=float, default=1)
    parser.add_argument("--p",             type=float, default=2)
    parser.add_argument("--n_critic",      type=int,   default=1)
    parser.add_argument("--reduce_fn",                 default="mean", choices=["mean", "sum", "max"])
    parser.add_argument("--reg",                       default="alp", choices=["gp", "lp", "alp"])
    args = parser.parse_args()
    print(args)

    # set GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # set seeds for reproducibility
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)

    sess = tf.InteractiveSession()

    run_name = str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_")
    log_dir = args.log_dir + str(int(args.ratio_plane)) + '/' + run_name
    if args.pretrained == None:
        os.makedirs(log_dir)

    reduce_fn = {
        "mean": tf.reduce_mean,
        "sum": tf.reduce_sum,
        "max": tf.reduce_max,
    }[args.reduce_fn]

    rho = tf.Variable(1/args.num_cluster * tf.ones(args.num_cluster))
    mu = [tf.Variable(tf.random_normal([args.latent_dim], stddev=0.1, seed=i), name=f'mu_{i}') for i in range(args.num_cluster)] # tf.zeros([args.latent_dim])
    sigma = [tf.Variable(1.0*tf.eye(args.latent_dim), name=f'sigma_{i}') for i in range(args.num_cluster)]

    with tf.name_scope('placeholders'):
        x_train_ph, y_train_ph = get_cifar10_tf(batch_size=args.batch_size, sess=sess)
        x_test_ph, y_test_ph = get_cifar10_tf(batch_size=args.val_size, sess=sess)
        x_10k_ph, y_10k_ph = get_cifar10_tf(batch_size=10000, sess=sess)
        x_50k_ph, y_50k_ph = get_cifar10_tf(batch_size=50000, sess=sess)

        global_step = tf.Variable(0, trainable=False, name='global_step')
        decay = tf.maximum(0., 1. - (tf.cast(global_step, tf.float32) / args.iterations))
        is_training = tf.placeholder(bool, name='is_training')
        use_agumentation = tf.identity(is_training, name='is_training')

    with tf.name_scope('pre_process'):
        x_train = (x_train_ph - 0.5) * 2.0
        x_test = (x_test_ph - 0.5) * 2.0

        x_true = x_train

        x_10k = (x_10k_ph - 0.5) * 2.0
        x_50k = (x_50k_ph - 0.5) * 2.0

    with tf.name_scope('gan'):
        pi = tf.nn.softmax(rho)
        components = []
        jitter = 1e-8
        for i in range(args.num_cluster):
            components.append(tfd.MultivariateNormalFullCovariance(loc=mu[i], covariance_matrix=sigma[i]))
        mvn = tfd.Mixture(cat=tfd.Categorical(probs=pi), components=components)

        z_gen = mvn.sample((args.batch_size * 2), name="z")
        x_generated = generator(z_gen, reuse=False)
        x_generated_ = DiffAugment(x_generated, policy='translation,color,cutout')

        d_true = discriminator(x_true, reuse=False)
        d_generated = discriminator(x_generated, reuse=True)

        z_reg = mvn.sample((tf.shape(x_true)[0]))
        x_reg = generator(z_reg, reuse=True)
        d_reg = discriminator(x_reg, reuse=True)

    with tf.name_scope('regularizer'):
        epsilon = tf.random_uniform([tf.shape(x_true)[0], 1, 1, 1], 0.0, 1.0)
        x_hat = epsilon * x_reg + (1 - epsilon) * x_true
        d_hat = discriminator(x_hat, reuse=True)

        gradients = tf.gradients(d_hat, x_hat)[0]

        dual_p = 1 / (1 - 1 / args.p) if args.p != 1 else np.inf
        gradient_norms = stable_norm(gradients, ord=dual_p)

        gp = gradient_norms - args.K
        gp_l2 = reduce_fn(gp ** 2)
        gp_loss = args.lambda_lp * gp_l2

        lp = tf.maximum(gradient_norms - args.K, 0)
        lp_l2 = reduce_fn(lp ** 2)
        lp_loss = args.lambda_lp * lp_l2

        d_regularizer_mean = tf.reduce_mean(tf.square(d_true))

    with tf.name_scope('alp'):
        samples = tf.concat([x_true, x_reg], axis=0)

        eps = args.eps_min + (args.eps_max - args.eps_min) * tf.random_uniform([tf.shape(samples)[0], 1, 1, 1], 0, 1)

        validity = discriminator(samples, reuse=True)

        d = tf.random_uniform(tf.shape(samples), 0, 1) - 0.5
        d = normalize(d, ord=2)
        for _ in range(args.ip):
            samples_hat = tf.clip_by_value(samples + args.xi * d, clip_value_min=-1, clip_value_max=1)
            validity_hat = discriminator(samples_hat, reuse=True)
            dist = tf.reduce_mean(tf.abs(validity - validity_hat))
            grad = tf.gradients(dist, [d], aggregation_method=2)[0]
            d = normalize(tf.stop_gradient(grad), ord=2)
        r_adv = d * eps

        samples_hat = tf.clip_by_value(samples + r_adv, clip_value_min=-1, clip_value_max=1)

        d_lp        = lambda x, x_hat: stable_norm(x - x_hat, ord=args.p)
        d_x         = d_lp

        samples_diff = d_x(samples, samples_hat)
        samples_diff = tf.maximum(samples_diff, 1e-10)

        validity      = discriminator(samples    , reuse=True)
        validity_hat  = discriminator(samples_hat, reuse=True)
        validity_diff = tf.abs(validity - validity_hat)

        alp = tf.maximum(validity_diff / samples_diff - args.K, 0)

        nonzeros = tf.greater(alp, 0)
        count = tf.reduce_sum(tf.cast(nonzeros, tf.float32))

        alp_l2 = reduce_fn(alp ** 2)
        alp_loss = args.lambda_lp * alp_l2

    with tf.name_scope('loss_gan'):
        wasserstein = (tf.reduce_mean(d_true) - tf.reduce_mean(d_reg))

        g_loss_ = -tf.reduce_mean(d_generated)
        d_loss = -wasserstein
        if args.reg == 'gp':
            d_loss += gp_loss
        elif args.reg == 'lp':
            d_loss += lp_loss
        elif args.reg == 'alp':
            d_loss += alp_loss

        ## contrastive loss
        responsibility_logit = tf.transpose([components[i].log_prob(z_gen) + tf.log(pi[i]) - mvn.log_prob(z_gen) for i in range(args.num_cluster)])

        cat_z = gumbel_softmax(responsibility_logit, 0.01, hard=False)
        enc_x = encoder(x_generated, args.latent_dim, False)
        I = tf.cast(args.iterations, tf.float32)
        X = tf.cast(global_step, tf.float32)
        temp = tf.maximum(0.0, args.start_temp + tf.cast(global_step, tf.float32) * (args.end_temp - args.start_temp) / (tf.cast(args.iterations, tf.float32)))

        mu_stack = tf.stack(mu)
        l_c = cosine_contrastive_loss(cat_z, mu_stack, enc_x, temp, args.scale, args.margin)

        enc_x_ = encoder(x_generated_, args.latent_dim, True)
        l_s = tf.reduce_mean(simclr_loss(enc_x, enc_x_))

        g_loss = 1 * g_loss_ + args.coeff * temp * tf.reduce_mean(l_c)
        g_loss_batch = 1 * (-d_generated) + args.coeff * temp * tf.expand_dims(l_c, 1)

        ## reparameterization trick
        responsibility = tf.nn.softmax(responsibility_logit)

        # Bonnet
        dldz = tf.gradients(g_loss_batch, z_gen)[0]
        gmu = tf.unstack(tf.reduce_mean(tf.expand_dims(responsibility, 2) * tf.expand_dims(dldz, 1), axis=0))

        # Price
        sigma_inv = tf.matrix_inverse(tf.stack(sigma))
        sigma_inv = 0.5 * (sigma_inv + tf.transpose(sigma_inv, [0, 2, 1]))
        S = tf.expand_dims(tf.expand_dims(responsibility, 2), 3) * tf.expand_dims(sigma_inv, 0) @ \
            tf.expand_dims((tf.expand_dims(z_gen, 1) - tf.expand_dims(tf.stack(mu), 0)), 3) @ \
            tf.expand_dims(tf.expand_dims(dldz, axis=1), 1)
        S = tf.reduce_mean(S, axis=0)
        nabla = 0.5 * (S + tf.transpose(S, [0, 2, 1]))
        reg_pd = 0.5 * args.lr_stein * decay * nabla @ sigma_inv @ nabla

        gsigma = nabla - reg_pd
        gsigma = tf.unstack(0.5 * (gsigma + tf.transpose(gsigma, [0, 2, 1])))

        entropy = tf.log(tf.linalg.det(sigma))
        ## Mixing coefficient
        grho = tf.reduce_mean((responsibility - pi) * (-d_generated), axis=0)

    # clustering metrics
    x_test_stack = array_ops.split(x_test, num_or_size_splits=10000 // 100)
    x_test_stack = array_ops.stack(x_test_stack)
    enc_x_test = map_fn(fn=partial(encoder, out_dim=args.latent_dim, reuse=True), elems=x_test_stack,
                        parallel_iterations=1, back_prop=False, swap_memory=True, dtype=tf.float32)
    enc_x_test = array_ops.concat(array_ops.unstack(enc_x_test), 0)
    prob_test = tf.nn.softmax(cosine_similarity_matrix(enc_x_test, mu_stack))

    with tf.name_scope('optimizer'):
        learning_rate = args.lr * decay
        g_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.0, beta2=0.9)
        d_optimizer = tf.train.AdamOptimizer(learning_rate=4 * learning_rate, beta1=0.0, beta2=0.9)
        e_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.0, beta2=0.9)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='gan/generator')
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        with tf.control_dependencies(update_ops):
            g_train = g_optimizer.minimize(g_loss, var_list=g_vars)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='gan/discriminator')
        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        with tf.control_dependencies(update_ops):
            d_train = d_optimizer.minimize(d_loss, var_list=d_vars, global_step=global_step)

        sgd_mu = tf.train.GradientDescentOptimizer(args.lr_stein * decay * 10)
        sgd_sigma = tf.train.GradientDescentOptimizer(args.lr_stein * decay)
        sgd_rho = tf.train.GradientDescentOptimizer(args.lr_stein * decay)

        opt_mu = sgd_mu.apply_gradients(zip(gmu, mu))
        opt_sigma = sgd_sigma.apply_gradients(zip(gsigma, sigma))
        opt_rho = sgd_rho.apply_gradients(zip([grho], [rho]))

        ## constastive
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='loss_gan/encoder')
        e_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
        with tf.control_dependencies(update_ops):
            e_train = e_optimizer.minimize(g_loss+l_s+1e-4*tf.add_n([tf.nn.l2_loss(v) for v in e_vars]), var_list=e_vars)

    with tf.name_scope('summaries'):
        scalars_summary('d_true', d_true)
        scalars_summary('d_generated', d_generated)
        scalars_summary('constrastive_loss', l_c)
        scalars_summary('simclr_loss', l_s)

        image_grid_summary('x_true', x_true)
        image_grid_summary('x_mixture', x_reg)
        image_grid_summary('gradients', gradients)

        for i in range(args.num_cluster):
            z_cond = components[i].sample((64))
            x_cond = generator(z_cond, reuse=True)
            image_grid_summary(f'x_cond_{i}', x_cond[:64])

        x_cond_stack = []
        for i in range(args.num_cluster):
            z_cond = components[i].sample((5000))
            z_cond_stack = array_ops.split(z_cond, num_or_size_splits=5000 // 100)
            z_cond_stack = array_ops.stack(z_cond_stack)
            x_cond = map_fn(fn=partial(generator, reuse=True), elems=z_cond_stack,
                            parallel_iterations=1, back_prop=False, swap_memory=True,
                            dtype=tf.float32)
            x_cond = array_ops.concat(array_ops.unstack(x_cond), 0)
            x_cond_stack.append(x_cond)

        merged_summary = tf.summary.merge_all()


        # Advanced metrics
        with tf.name_scope('validation'):
            # INCEPTION VALIDATION
            # Specific function to compute inception score for very large number of samples
            def generate_resize_and_classify(z):
                INCEPTION_OUTPUT = 'logits:0'
                x = generator(z, reuse=True)
                x = tf.image.resize_bilinear(x, [299, 299])
                return tf.contrib.gan.eval.run_inception(x, output_tensor=INCEPTION_OUTPUT)

            # Fixed z for fairness between runs
            inception_z = mvn.sample((10000))
            inception_score = tf.contrib.gan.eval.classifier_score(
                inception_z,
                classifier_fn=generate_resize_and_classify,
                num_batches=10000 // 100
            )

            inception_summary = tf.summary.merge([tf.summary.scalar('inception_score', inception_score)])

            # FID VALIDATION
            def resize_and_classify(x):
                INCEPTION_FINAL_POOL = 'pool_3:0'
                x = tf.image.resize_bilinear(x, [299, 299])
                return tf.contrib.gan.eval.run_inception(x, output_tensor=INCEPTION_FINAL_POOL)

            fid_real = x_10k
            fid_z = mvn.sample((10000))
            fid_z_list = array_ops.split(fid_z, num_or_size_splits=10000 // 100)
            fid_z_batches = array_ops.stack(fid_z_list)
            fid_gen = map_fn(
                fn=partial(generator, reuse=True),
                elems=fid_z_batches,
                parallel_iterations=1,
                back_prop=False,
                swap_memory=True,
                name='RunGenerator',
                dtype=tf.float32
            )
            fid_gen = array_ops.concat(array_ops.unstack(fid_gen), 0)
            fid = tf.contrib.gan.eval.frechet_classifier_distance(
                fid_real,
                fid_gen,
                classifier_fn=resize_and_classify,
                num_batches=10000 // 100
            )

            fid_summary = tf.summary.merge([tf.summary.scalar('fid', fid)])

            full_summary = tf.summary.merge([merged_summary, inception_summary, fid_summary])

        # Final eval
        with tf.name_scope('test'):
            # INCEPTION TEST
            # Specific function to compute inception score for very large number of samples
            def generate_resize_and_classify(z):
                INCEPTION_OUTPUT = 'logits:0'
                x = generator(z, reuse=True)
                x = tf.image.resize_bilinear(x, [299, 299])
                return tf.contrib.gan.eval.run_inception(x, output_tensor=INCEPTION_OUTPUT)


            # Fixed z for fairness between runs
            inception_z_final = mvn.sample((100000))
            inception_score_final = tf.contrib.gan.eval.classifier_score(
                inception_z_final,
                classifier_fn=generate_resize_and_classify,
                num_batches=100000 // 100
            )

            inception_summary_final = tf.summary.merge([tf.summary.scalar('inception_score_final', inception_score_final)])

            # FID TEST
            def resize_and_classify(x):
                INCEPTION_FINAL_POOL = 'pool_3:0'
                x = tf.image.resize_bilinear(x, [299, 299])
                return tf.contrib.gan.eval.run_inception(x, output_tensor=INCEPTION_FINAL_POOL)

            fid_real_final = x_10k
            fid_z_final = mvn.sample((10000))
            fid_z_final_list = array_ops.split(fid_z_final, num_or_size_splits=10000 // 100)
            fid_z_final_batches = array_ops.stack(fid_z_final_list)
            fid_gen_final = map_fn(
                fn=partial(generator, reuse=True),
                elems=fid_z_final_batches,
                parallel_iterations=1,
                back_prop=False,
                swap_memory=True,
                name='RunGenerator',
                dtype=tf.float32
            )
            fid_gen_final = array_ops.concat(array_ops.unstack(fid_gen_final), 0)
            fid_final = tf.contrib.gan.eval.frechet_classifier_distance(
                fid_real_final,
                fid_gen_final,
                classifier_fn=resize_and_classify,
                num_batches=10000 // 100
            )

            fid_summary_final = tf.summary.merge([tf.summary.scalar('fid_final', fid_final)])
            final_summary = tf.summary.merge([merged_summary, inception_summary_final, fid_summary_final])

        summary_writer = tf.summary.FileWriter(log_dir)

    # Initialize all TF variables
    sess.run([
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    ])

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # load weight
    saver = tf.train.Saver(max_to_keep=1000)
    if args.pretrained != None:
        saver.restore(sess, f'{args.pretrained}')
        print(f'model loaded! pretrained model path: {args.pretrained}')

    # Standardized validation z
    z_validate = np.random.randn(args.val_size, args.latent_dim)

    print(f"Logging to: {log_dir}")

    evals = [d_train, d_loss]

    # Train the network
    t = tqdm(range(args.iterations))
    i = sess.run(global_step)

    if args.pretrained == None:
        for _ in t:
            for j in range(args.n_critic):
                results = sess.run(
                    evals,
                    feed_dict={is_training: True}
                )
                d_loss_result = results[1]
            g_loss_result, _, l2c_result, _, _, _, _ = sess.run([g_loss_, g_train, l_c, e_train, opt_mu, opt_sigma, opt_rho], feed_dict={is_training: True})

            if i % args.log_freq == args.log_freq - 1:
                merged_summary_result_train = sess.run(merged_summary, feed_dict={is_training: False})
                summary_writer.add_summary(merged_summary_result_train, i)
            if i % args.val_freq == args.val_freq - 1:
                merged_summary_result_test = sess.run(full_summary, feed_dict={is_training: False})
                summary_writer.add_summary(merged_summary_result_test, i)


            if i % args.save_freq == args.save_freq - 1:
                saver.save(sess, f'{log_dir}/model', i+1)

            t.set_description( f"[D loss: {d_loss_result:.4f}] [G loss: {g_loss_result:.4f}] [Lc: {np.mean(l2c_result):.8f}]")

            if (i + 1) == args.iterations:
                print('Measuring unsupervised conditional generation metric...')
                x_real = []
                x_fake, x_real_, y_test_ = sess.run([x_cond_stack, x_50k, y_50k_ph], {is_training: False})
                for j in range(args.num_cluster):
                    x_real.append((x_real_[y_test_.argmax(-1) == j] + 1) * 255 / 2)
                x_fake = (np.array(x_fake) + 1) * 255 / 2
                icfid, best_idx = calculate_cluster_distance(x_real, x_fake, args.num_cluster, sess)
                print(f'\nICFID:{np.mean(icfid)}'
                      f'\nBest_idx:{*best_idx,}')
                summary = tf.Summary()
                tags_icfid = [f'ICFID/cluster_{j}' for j in range(args.num_cluster)]
                summary.value.add(tag='ICFID/mean', simple_value=np.mean(icfid))
                for j, value in enumerate(icfid):
                    summary.value.add(tag=tags_icfid[j], simple_value=value)
                tags_idx = [f'best_idx/cluster_{j}' for j in range(args.num_cluster)]
                for j, value in enumerate(best_idx):
                    summary.value.add(tag=tags_idx[j], simple_value=value)
                print('Measuring unconditional generation metrics...')
                is_final_, fid_final_ = sess.run([inception_score_final, fid_final], {is_training: False})
                summary.value.add(tag='summaries/IS', simple_value=is_final_)
                summary.value.add(tag='summaries/FID', simple_value=fid_final_)
                print(f'IS:{is_final_}'
                      f'\nFID:{fid_final_}')
                summary_writer.add_summary(summary, i)
                summary_writer.flush()

            if (i + 1) % args.log_freq == 0:
                # mixing coefficient
                pi_ = sess.run(pi, feed_dict={is_training: False})
                print(f'\npi:{*pi_,}')
                # clustering metrics
                enc_x_test_, y_test_ = sess.run([prob_test, y_test_ph], feed_dict={is_training: False})
                ari_r, nmi_r = calc_metrics(enc_x_test_, y_test_, args.num_cluster)

                summary = tf.Summary()
                tags = ['cluster_assignment/ARI', 'cluster_assignment/NMI']
                values = [ari_r, nmi_r]
                for j, value in enumerate(values):
                    summary.value.add(tag=tags[j], simple_value=value)
                for j in range(args.num_cluster):
                    summary.value.add(tag=f'pi/cluster_{j}', simple_value=pi_[j])
                summary_writer.add_summary(summary, i)
                summary_writer.flush()
                print(f'[ARI: {ari_r:.4f}] [NMI: {nmi_r:.4f}]')

            i += 1

    else:
        # ICFID
        print('Measuring unsupervised conditional generation metric...')
        x_real = []
        x_fake, x_real_, y_test_ = sess.run([x_cond_stack, x_50k, y_50k_ph], {is_training: False})
        for j in range(args.num_cluster):
            x_real.append((x_real_[y_test_.argmax(-1) == j] + 1) * 255 / 2)
        x_fake = (np.array(x_fake) + 1) * 255 / 2
        icfid, best_idx = calculate_cluster_distance(x_real, x_fake, args.num_cluster, sess)
        print(f'\nICFID:{np.mean(icfid)}'
              f'\nBest_idx:{*best_idx,}')

        # IS and FID
        print('Measuring unconditional generation metrics...')
        is_final_, fid_final_ = sess.run([inception_score_final, fid_final], {is_training: False})
        print(f'IS:{is_final_}'
              f'\nFID:{fid_final_}')

        # mixing coefficient
        pi_ = sess.run(pi, feed_dict={is_training: False})
        fmtstr = '{:-3d}  ' * args.num_cluster
        print(f'pi:{*pi_,}')

        # clustering metrics
        print('Measuring clustering metrics...')
        enc_x_test_, y_test_ = sess.run([prob_test, y_test_ph], feed_dict={is_training: False})
        ari_r, nmi_r = calc_metrics(enc_x_test_, y_test_, args.num_cluster)
        print(f'[ARI: {ari_r:.4f}] [NMI: {nmi_r:.4f}]')

    if args.pretrained == None:
        print(f"Logging to: {log_dir}")
