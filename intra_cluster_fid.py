import numpy as np
import os
import tensorflow as tf
from scipy import linalg
import pathlib
import warnings
from tqdm import tqdm

cur_dirname = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = '%s/res/' % cur_dirname

class InvalidFIDException(Exception):
    pass


def create_inception_graph(pth):
    with tf.gfile.FastGFile(pth, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='FID_Inception_Net')

def check_or_download_inception(inception_path):
    INCEPTION_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    if inception_path is None:
        inception_path = MODEL_DIR
    inception_path = pathlib.Path(inception_path)
    model_file = inception_path / 'classify_image_graph_def.pb'
    if not model_file.exists():
        print("Downloading Inception model")
        from urllib import request
        import tarfile
        fn, _ = request.urlretrieve(INCEPTION_URL)
        with tarfile.open(fn, mode='r') as f:
            f.extract('classify_image_graph_def.pb', str(model_file.parent))
    return str(model_file)

def _get_inception_layer(sess):
    layername = 'FID_Inception_Net/pool_3:0'
    pool3 = sess.graph.get_tensor_by_name(layername)
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims != [] and (shape._dims is not None):
                shape = [s.value for s in shape]
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o.__dict__['_shape_val'] = tf.TensorShape(new_shape)
    return pool3

def get_activations(images, sess, batch_size=50, verbose=False):
    inception_layer = _get_inception_layer(sess)
    d0 = images.shape[0]
    if batch_size > d0:
        print("warning: batch size is bigger than the data size. setting batch size to data size")
        batch_size = d0
    n_batches = d0 // batch_size
    n_used_imgs = n_batches * batch_size
    pred_arr = np.empty((n_used_imgs, 2048))
    for i in range(n_batches):
        if verbose:
            print("\rPropagating batch %d/%d" % (i + 1, n_batches))
        start = i * batch_size
        end = start + batch_size
        batch = images[start:end]
        pred = sess.run(inception_layer, {'FID_Inception_Net/ExpandDims:0': batch})
        pred_arr[start:end] = pred.reshape(batch_size, -1)
    if verbose:
        print(" done")
    return pred_arr

def calculate_activation_statistics(images, sess, batch_size=50, verbose=False):
    act = get_activations(images, sess, batch_size, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def greedy_icfid(mat):
    best_idx = []
    icfid = []
    for i in range(len(mat)):
        idx = np.argmin(mat[:,i])
        best_idx.append(idx)
        icfid.append(mat[idx,i])
        mat[idx,:] = 999999
    return icfid, best_idx

def calculate_cluster_distance(x_real, x_fake, num_class, sess, eps=1e-6):
    inception_path = check_or_download_inception(None)
    create_inception_graph(str(inception_path))

    mu1 = []
    mu2 = []
    sigma1 = []
    sigma2 = []
    for i in range(num_class):
        m_r, s_r = calculate_activation_statistics(x_real[i], sess)
        m_f, s_f = calculate_activation_statistics(x_fake[i], sess)
        mu1.append(m_r)
        mu2.append(m_f)
        sigma1.append(s_r)
        sigma2.append(s_f)

    fid_mat = np.zeros([num_class, num_class])
    for i in range(num_class):
        for j in range(num_class):
            print(f'{i}th fake & {j}th real')
            diff = mu1[j] - mu2[i]

            # product might be almost singular
            covmean, _ = linalg.sqrtm(sigma1[j].dot(sigma2[i]), disp=False)
            if not np.isfinite(covmean).all():
                msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
                warnings.warn(msg)
                offset = np.eye(sigma1[j].shape[0]) * eps
                covmean = linalg.sqrtm((sigma1[j] + offset).dot(sigma2[i] + offset))

            # numerical error might give slight imaginary component
            if np.iscomplexobj(covmean):
                if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                    m = np.max(np.abs(covmean.imag))
                    raise ValueError("Imaginary component {}".format(m))
                covmean = covmean.real

            tr_covmean = np.trace(covmean)

            fid_mat[i,j] = diff.dot(diff) + np.trace(sigma1[j]) + np.trace(sigma2[i]) - 2 * tr_covmean
    icfid, best_idx = greedy_icfid(fid_mat)
    return icfid, best_idx
