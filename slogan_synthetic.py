import argparse
import datetime
import os
import numpy as np
import math
from tqdm import tqdm
import tensorflow_probability as tfp
import tensorflow as tf
import sklearn.datasets
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans

tfd = tfp.distributions
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
map_fn = tf.map_fn

# set seeds for reproducibility
np.random.seed(0)
tf.set_random_seed(0)

# DATA
class DataSet(object):
    def __init__(self, data, label):
        assert data.ndim == 2

        self.num_examples = data.shape[0]

        self.data = data
        self.label = label

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
            self.data = self.data[perm]
            self.label = self.label[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch
        return self.data[start:end], self.label[start:end]

def get_toy_dataset():
    data = np.empty((0,2))
    label = np.empty((0,8))
    ratio = [1/16, 3/16, 1/16, 3/16, 1/16, 3/16, 1/16, 3/16]
    mean = [[2,0],[-2,0],[0,2],[0,-2],[np.sqrt(2),np.sqrt(2)],[-np.sqrt(2),-np.sqrt(2)],[np.sqrt(2),-np.sqrt(2)],[-np.sqrt(2),np.sqrt(2)]]

    for i in range(8):
        data = np.vstack((data, np.random.multivariate_normal(mean[i], 0.01 * np.eye(2), int(80000 * ratio[i]))))
        label = np.vstack((label, np.tile(np.eye(8)[i], [int(80000 * ratio[i]), 1])))
    scaler = sklearn.preprocessing.MinMaxScaler((-1,1))
    data = scaler.fit_transform(data).astype('float32')

    perm = np.arange(80000)
    np.random.shuffle(perm)
    data = data[perm]
    label = label[perm]

    dataset = DataSet(data, label)

    return dataset


def get_toy_tf(sess, batch_size=1, split=None, start_queue_runner=True):
    with tf.name_scope('get_toy_tf'):
        dataset = get_toy_dataset()
        data_placeholder = tf.placeholder(dataset.data.dtype, dataset.data.shape)
        label_placeholder = tf.placeholder(dataset.label.dtype, dataset.label.shape)
        data = tf.data.Dataset.from_tensor_slices((data_placeholder, label_placeholder))
        data = data.repeat(None)
        data = data.prefetch(tf.data.experimental.AUTOTUNE)
        data = data.shuffle(buffer_size=100000)
        data = data.batch(batch_size)
        iterator = data.make_initializable_iterator()
        data_batch = iterator.get_next()
        sess.run(iterator.initializer, feed_dict={data_placeholder: dataset.data, label_placeholder: dataset.label})

        if start_queue_runner:
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord)

        return data_batch


def bn(x, is_training):
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

def l2_normalize(x, eps=1e-12):
    return x / tf.linalg.norm(x + eps)


class Spectral_Norm(tf.keras.constraints.Constraint):
    def __init__(self, power_iters=2):
        self.n_iters = power_iters

    def __call__(self, w):
      flattened_w = tf.reshape(w, [w.shape[0], -1])
      u = tf.random.normal([tf.shape(flattened_w)[0]])
      v = tf.random.normal([tf.shape(flattened_w)[1]])
      for i in range(self.n_iters):
        v = tf.linalg.matvec(tf.transpose(flattened_w), u)
        v = l2_normalize(v)
        u = tf.linalg.matvec(flattened_w, v)
        u = l2_normalize(u)
      sigma = tf.tensordot(u, tf.linalg.matvec(flattened_w, v), axes=1)
      return w / sigma

    def get_config(self):
        return {'n_iters': self.n_iters}


def generator(z, is_training, reuse):
    with tf.variable_scope('generator', reuse=reuse):
        x = tf.layers.dense(z, 128)
        x = tf.nn.relu(bn(x, is_training))
        x = tf.layers.dense(x, 128)
        x = tf.nn.relu(bn(x, is_training))
        result = tf.layers.dense(x, 2)
        return tf.tanh(result)

def discriminator(x, reuse):
    with tf.variable_scope('discriminator', reuse=reuse):
        x = tf.layers.dense(x, 128)
        x = tf.nn.leaky_relu(x)
        x = tf.layers.dense(x, 128)
        x = tf.nn.leaky_relu(x)
        flat = tf.layers.dense(x, 1)
        return flat

def encoder(x, dim, reuse):
    with tf.variable_scope('encoder', reuse=reuse):
        x = tf.layers.dense(x, 128, kernel_constraint=Spectral_Norm())
        x = tf.nn.leaky_relu(x)
        x = tf.layers.dense(x, 128, kernel_constraint=Spectral_Norm())
        x = tf.nn.leaky_relu(x)
        flat = tf.layers.dense(x, dim, kernel_constraint=Spectral_Norm())
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
    sin_t2 = tf.maximum(tf.subtract(1., cos_t2, name='sin_2'), 1e-8)
    sin_t = tf.sqrt(sin_t2+1e-8, name='sin_t')
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
        denominator = tf.reduce_mean(tf.concat([tf.expand_dims(tf.exp(sim_enc_mu),-1), remove_diag(tf.exp(sim_enc_mu_mat))], 1), 1)
        criterion = -tf.log(numerator/denominator)
        return criterion

## evaluation metrics for clustering
def compute_purity(y_pred, y_true):
    clusters = set(y_pred)
    correct = 0
    for cluster in clusters:
        indices = np.where(y_pred == cluster)[0]

        cluster_labels = y_true[indices]
        majority_label = np.argmax(np.bincount(cluster_labels))
        correct += np.sum(cluster_labels == majority_label)
    return float(correct) / len(y_pred)

def calc_metrics(enc_x, y_test, num_cluster):
    km = KMeans(n_clusters=max(num_cluster, len(np.unique(y_test.argmax(axis=-1)))), random_state=0).fit(enc_x)
    labels_pred = km.labels_

    purity = compute_purity(labels_pred, y_test.argmax(axis=-1))
    ari = metrics.adjusted_rand_score(y_test.argmax(axis=-1), labels_pred)
    nmi = metrics.normalized_mutual_info_score(y_test.argmax(axis=-1), labels_pred)
    return purity, ari, nmi


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir",                   default="./logs/synthetic/")
    parser.add_argument("--gpu",           type=str,   default='3')
    parser.add_argument("--ckpt_path",                 default=None)
    parser.add_argument("--num_cluster",   type=int,   default=8)
    parser.add_argument("--lr_stein",      type=float, default=1e-2)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--coeff",         type=float, default=4)
    parser.add_argument("--scale",         type=float, default=2.0)
    parser.add_argument("--margin",        type=float, default=0.5)
    parser.add_argument("--start_temp",    type=float, default=1.0)
    parser.add_argument("--end_temp",      type=float, default=-4.0)
    parser.add_argument("--log_freq",      type=int,   default=100)
    parser.add_argument("--iterations",    type=int,   default=100000)
    parser.add_argument("--batch_size",    type=int,   default=128)
    parser.add_argument("--save_freq",     type=int,   default=1000)
    parser.add_argument("--val_size",      type=int,   default=100)
    parser.add_argument("--random_seed",   type=int,   default=0)
    parser.add_argument("--latent_dim",    type=int,   default=64)
    parser.add_argument("--lambda_lp",     type=float, default=10)
    parser.add_argument("--K",             type=float, default=1)
    parser.add_argument("--p",             type=float, default=2)
    parser.add_argument("--n_critic",      type=int,   default=1)
    parser.add_argument("--reduce_fn",     default="mean", choices=["mean", "sum", "max"])
    parser.add_argument("--reg",           default="lp",   choices=["gp", "lp"])
    args = parser.parse_args()
    print(args)

    # set GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # set seeds for reproducibility
    np.random.seed(0)
    tf.set_random_seed(0)

    sess = tf.InteractiveSession()

    run_name = str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_")
    log_dir = args.log_dir + run_name
    os.makedirs(log_dir)

    reduce_fn = {
        "mean": tf.reduce_mean,
        "sum": tf.reduce_sum,
        "max": tf.reduce_max,
    }[args.reduce_fn]

    ##
    rho = tf.Variable(1/args.num_cluster * tf.ones(args.num_cluster))
    mu = [tf.Variable(tf.random_normal([args.latent_dim], stddev=0.1, seed=i), name=f'mu_{i}') for i in range(args.num_cluster)]
    sigma = [tf.Variable(1.0*tf.eye(args.latent_dim), name=f'sigma_{i}') for i in range(args.num_cluster)]

    with tf.name_scope('placeholders'):
        x_train, y_train = get_toy_tf(batch_size=args.batch_size, sess=sess)
        x_10k_ph, y_10k_ph = get_toy_tf(batch_size=10000, sess=sess)

        global_step = tf.Variable(0, trainable=False, name='global_step')
        decay = tf.maximum(0., 1. - (tf.cast(global_step, tf.float32) / args.iterations))
        is_training = tf.placeholder(bool, name='is_training')
        use_agumentation = tf.identity(is_training, name='is_training')

    with tf.name_scope('gan'):
        pi = tf.nn.softmax(rho)
        components = []
        jitter = 1e-8
        for i in range(args.num_cluster):
            components.append(tfd.MultivariateNormalFullCovariance(loc=mu[i], covariance_matrix=sigma[i]))
        mvn = tfd.Mixture(cat=tfd.Categorical(probs=pi), components=components)

        z_gen = mvn.sample((args.batch_size), name="z")
        x_generated = generator(z_gen, is_training, reuse=False)

        d_true = discriminator(x_train, reuse=False)
        d_generated = discriminator(x_generated, reuse=True)


    with tf.name_scope('regularizer'):
        epsilon = tf.random_uniform([tf.shape(x_train)[0], 1], 0.0, 1.0)
        x_hat = epsilon * x_generated + (1 - epsilon) * x_train
        d_hat = discriminator(x_hat, reuse=True)
        gradients = tf.gradients(d_hat, x_hat)[0]

        dual_p = 1 / (1 - 1 / args.p) if args.p != 1 else np.inf
        gradient_norms = stable_norm(gradients, ord=dual_p)

        lp = tf.maximum(gradient_norms - args.K, 0)
        lp_l2 = reduce_fn(lp ** 2)
        lp_loss = args.lambda_lp * lp_l2

        d_regularizer_mean = tf.reduce_mean(tf.square(d_true))


    with tf.name_scope('loss_gan'):
        wasserstein = tf.reduce_mean(tf.minimum(0.0, -1+d_true)) + tf.reduce_mean(tf.minimum(0.0, -1-d_generated))
        g_loss_ = -tf.reduce_mean(d_generated)
        d_loss = -wasserstein
        if args.reg == 'lp':
            d_loss += lp_loss

        ## contrastive loss
        responsibility_logit = tf.transpose([components[i].log_prob(z_gen)+tf.log(pi[i])-mvn.log_prob(z_gen) for i in range(args.num_cluster)])

        cat_z = gumbel_softmax(responsibility_logit, 0.1, hard=False)
        enc_x = encoder(x_generated, args.latent_dim, False)
        temp = tf.maximum(0.0, args.start_temp + tf.cast(global_step, tf.float32) * (args.end_temp - args.start_temp) / (tf.cast(args.iterations, tf.float32)))

        mu_stack = tf.stack(mu)
        Lc = cosine_contrastive_loss(cat_z, mu_stack, enc_x, temp, args.scale, args.margin)

        g_loss = g_loss_ + args.coeff*temp * tf.reduce_mean(Lc)
        g_loss_batch = (-d_generated) + args.coeff*temp * tf.expand_dims(Lc, 1)

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
        enc_x_test = encoder(x_10k_ph, args.latent_dim, True)

    with tf.name_scope('optimizer'):
        learning_rate = args.lr * decay
        g_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0., beta2=0.9)
        d_optimizer = tf.train.AdamOptimizer(learning_rate=4*learning_rate, beta1=0., beta2=0.9)
        e_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0., beta2=0.9)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='gan/generator')
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        with tf.control_dependencies(update_ops):
            g_train = tf.train.experimental.enable_mixed_precision_graph_rewrite(g_optimizer).minimize(g_loss, var_list=g_vars)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='gan/discriminator')
        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        with tf.control_dependencies(update_ops):
            d_train = tf.train.experimental.enable_mixed_precision_graph_rewrite(d_optimizer).minimize(d_loss, var_list=d_vars, global_step=global_step)

        sgd_mu = tf.train.GradientDescentOptimizer(args.lr_stein * decay * 10)
        sgd_sigma = tf.train.GradientDescentOptimizer(args.lr_stein * decay)
        sgd_rho = tf.train.GradientDescentOptimizer(args.lr_stein * decay)

        opt_mu = tf.train.experimental.enable_mixed_precision_graph_rewrite(sgd_mu).apply_gradients(zip(gmu, mu))
        opt_sigma = tf.train.experimental.enable_mixed_precision_graph_rewrite(sgd_sigma).apply_gradients(zip(gsigma, sigma))
        opt_rho = tf.train.experimental.enable_mixed_precision_graph_rewrite(sgd_rho).apply_gradients(zip([grho], [rho]))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='loss_gan/encoder')
        e_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
        with tf.control_dependencies(update_ops):
            e_train = tf.train.experimental.enable_mixed_precision_graph_rewrite(e_optimizer).minimize(g_loss, var_list=e_vars)

    with tf.name_scope('summaries'):
        x_cond = []
        x_mu = []
        z_cond_10k = mvn.sample((80000))
        z_cond = mvn.sample((8000))
        x_cond_sorted = generator(z_cond, is_training, reuse=True)
        x_cond.append(x_cond_sorted)
        for i in range(args.num_cluster):
            z_cond = components[i].sample((tf.cast(80000*pi[i], tf.int32)))
            z_cond_idx = tf.argsort(components[i].log_prob(z_cond), direction='DESCENDING')
            z_cond_sorted = tf.gather(z_cond, z_cond_idx)
            x_cond_sorted = generator(z_cond_sorted, is_training, reuse=True)
            x_cond.append(x_cond_sorted)
            # x_mu
            x_mu_cluster = generator(tf.reshape(mu[i],[1,-1]), is_training, reuse=True)
            x_mu.append(x_mu_cluster)

    # Initialize all TF variables
    sess.run([
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    ])

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    print(f"Logging to: {log_dir}")

    evals = [d_train, d_loss]
    # Train the network
    t = tqdm(range(args.iterations))
    i = sess.run(global_step)

    color = ['gray', 'red', 'orange', 'green', 'blue', 'purple', 'saddlebrown', 'greenyellow', 'cyan', 'darkkhaki', 'salmon', ]
    x_true_ = sess.run(x_10k_ph)
    plt.scatter(x_true_[:, 0], x_true_[:, 1], color='black', alpha=0.05, s=5)
    plt.savefig(f'{log_dir}/real_data.png')
    plt.close()

    for _ in t:
        if args.ckpt_path == None:
            for j in range(args.n_critic):
                _, d_loss_result = sess.run([d_train, d_loss], feed_dict={is_training: True})
            g_loss_result, _, Lc_result, _, _, _, _ = sess.run([g_loss_, g_train, Lc, e_train, opt_mu, opt_sigma, opt_rho], feed_dict={is_training: True})
        else:
            d_loss_result, g_loss_result = sess.run([d_loss, g_loss_], feed_dict={is_training: True})

        if i % args.log_freq == args.log_freq - 1:
            # figure
            x_cond_, x_mu_ = sess.run([x_cond, x_mu], feed_dict={is_training: False})
            plt.scatter(x_true_[:, 0], x_true_[:, 1], color='gainsboro', s=5)
            for j in range(args.num_cluster+1):
                plt.scatter(x_cond_[j][:, 0], x_cond_[j][:, 1], color=color[j], alpha=0.75, s=5)
            for j in range(args.num_cluster):
                plt.scatter(x_mu_[j][:, 0], x_mu_[j][:, 1], color=color[j+1], edgecolors='black', s=100, linewidth=3)
            plt.savefig(f'{log_dir}/samples_{i+1}.png')
            plt.close()

        t.set_description(f"[D loss: {d_loss_result:.4f}] [G loss: {g_loss_result:.4f}] [Lc: {np.mean(Lc_result):.8f}]")

        if i % 1000 == 0:
            # clustering metrics
            pi_ = sess.run(pi, feed_dict={is_training: False})
            enc_x_test_, y_test_ = sess.run([enc_x_test, y_10k_ph], feed_dict={is_training: False})
            purity_r, ari_r, nmi_r = calc_metrics(enc_x_test_, y_test_, args.num_cluster)

            fmtstr = '{:-3d}  ' * args.num_cluster
            print(f'\npi:{*pi_,}'
                  f'\n[purity: {purity_r:.4f}] [ari: {ari_r:.4f}] [nmi: {nmi_r:.4f}]')
        i += 1