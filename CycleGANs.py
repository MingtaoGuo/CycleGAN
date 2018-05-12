import tensorflow as tf
import numpy as np
import scipy.misc as misc
from PIL import Image
import os



epsilon = 1e-8
img_W = 128
img_H = 128
img_nums = 8000
batchsize = 5


def conv(inputs, nums_out, ksize, stride, padding, is_dis=False):
    c = int(inputs.shape[-1])
    W = tf.get_variable("W", shape=[ksize, ksize, c, nums_out], initializer=tf.random_normal_initializer(stddev=0.02))
    b = tf.get_variable("b", shape=[nums_out], initializer=tf.constant_initializer([0]))
    if is_dis:
        return tf.nn.conv2d(inputs, spectral_norm("SN",W), [1, stride, stride, 1], padding) + b
    else:
        return tf.nn.conv2d(inputs, W, [1, stride, stride, 1], padding) + b

def deconv(inputs, nums_out, ksize, stride):
    c = int(inputs.shape[-1])
    batch = int(inputs.shape[0])
    height = int(inputs.shape[1])
    width = int(inputs.shape[2])
    W = tf.get_variable("W", shape=[ksize, ksize, nums_out, c], initializer=tf.random_normal_initializer(stddev=0.02))
    b = tf.get_variable("b", shape=[nums_out], initializer=tf.constant_initializer([0.]))
    return tf.nn.conv2d_transpose(inputs, W, output_shape=[batch, height*stride, width*stride, nums_out], strides=[1, stride, stride, 1]) + b

def InstanceNorm(inputs):
    mean, var = tf.nn.moments(inputs, axes=[1, 2], keep_dims=True)
    scale = tf.get_variable("scale", shape=mean.shape, initializer=tf.constant_initializer([1.]))
    shift = tf.get_variable("shift", shape=mean.shape, initializer=tf.constant_initializer([0.]))
    return (inputs - mean) * scale / tf.sqrt(var + epsilon) + shift

def leaky_relu(inputs, slope=0.2):
    return tf.maximum(slope*inputs, inputs)

def spectral_norm(name, w, iteration=1):
    #Spectral normalization which was published on ICLR2018,please refer to "https://www.researchgate.net/publication/318572189_Spectral_Normalization_for_Generative_Adversarial_Networks"
    #This function spectral_norm is forked from "https://github.com/taki0112/Spectral_Normalization-Tensorflow"
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])
    with tf.variable_scope(name, reuse=False):
        u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
    u_hat = u
    v_hat = None

    def l2_norm(v, eps=1e-12):
        return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

    for i in range(iteration):
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)
        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)
    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma
    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)
    return w_norm


class discriminator:
    def __init__(self, name):
        self.name = name

    def __call__(self, inputs, reuse = False):
        #Patch discriminator
        inputs = tf.random_crop(inputs, [batchsize, 70, 70, 3])
        with tf.variable_scope(self.name, reuse=reuse):
            with tf.variable_scope("c64"):
                inputs = leaky_relu(conv(inputs, 64, 5, 2, "SAME", True))
            with tf.variable_scope("c128"):
                inputs = leaky_relu(InstanceNorm(conv(inputs, 128, 5, 2, "SAME", True)))
            with tf.variable_scope("c256"):
                inputs = leaky_relu(InstanceNorm(conv(inputs, 256, 5, 2, "SAME", True)))
            with tf.variable_scope("c512"):
                inputs = leaky_relu(InstanceNorm(conv(inputs, 512, 5, 2, "SAME", True)))
            with tf.variable_scope("fully_conv"):
                ksize = np.size(inputs, 1)
                inputs = tf.squeeze(conv(inputs, 1, ksize, 1, "VALID", True), axis=[1, 2, 3])
        return inputs

    @property
    def var(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)


class generator:
    def __init__(self, name):
        self.name = name

    def __call__(self, inputs, reuse=False):
        with tf.variable_scope(name_or_scope=self.name, reuse=reuse):
            inputs = tf.pad(inputs, tf.constant([[0, 0], [3, 3], [3, 3], [0, 0]]))
            with tf.variable_scope("c7s1-32"):
                inputs = tf.nn.relu(InstanceNorm(conv(inputs, 32, 7, 1, "VALID")))
            with tf.variable_scope("d64"):
                inputs = tf.nn.relu(InstanceNorm(conv(inputs, 64, 3, 2, "SAME")))
            with tf.variable_scope("d128"):
                inputs = tf.nn.relu(InstanceNorm(conv(inputs, 128, 3, 2, "SAME")))
            for i in range(6):
                with tf.variable_scope("R"+str(i)):
                    temp = inputs
                    with tf.variable_scope("R_conv1"):
                        inputs = tf.pad(inputs, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]), "REFLECT")
                        inputs = tf.nn.relu(InstanceNorm(conv(inputs, 128, 3, 1, "VALID")))
                    with tf.variable_scope("R_conv2"):
                        inputs = tf.pad(inputs, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]), "REFLECT")
                        inputs = InstanceNorm(conv(inputs, 128, 3, 1, "VALID"))
                    inputs = temp + inputs
            with tf.variable_scope("u64"):
                inputs = tf.nn.relu(InstanceNorm(deconv(inputs, 64, 3, 2)))
            with tf.variable_scope("u32"):
                inputs = tf.nn.relu(InstanceNorm(deconv(inputs, 32, 3, 2)))
            with tf.variable_scope("c7s1-3"):
                inputs = tf.nn.tanh((deconv(inputs, 3, 7, 1)))
            return (inputs + 1.) * 127.5

    @property
    def var(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

class CycleGAN:
    def __init__(self):
        self.X = tf.placeholder("float", shape=[batchsize, img_H, img_W, 3])
        self.Y = tf.placeholder("float", shape=[batchsize, img_H, img_W, 3])
        G = generator("G")
        F = generator("F")
        self.Dx = discriminator("Dx")
        self.Dy = discriminator("Dy")
        self.fake_X = F(self.Y)
        self.fake_Y = G(self.X)
        self.logits_real_X = self.Dx(self.X)
        self.logits_real_Y = self.Dy(self.Y)
        self.logits_fake_X = self.Dx(self.fake_X, True)
        self.logits_fake_Y = self.Dy(self.fake_Y, True)
        self.L_cyc = tf.reduce_mean(tf.abs(F(self.fake_Y, True) - self.X)) + tf.reduce_mean(tf.abs(G(self.fake_X, True) - self.Y))
        #WGAN's Loss function is used here, which is different from the paper CycleGAN where used LSGAN's loss function
        #WGAN has been proved that it can yield high qulity result and make the training process more stable
        self.d_loss_Y = -tf.reduce_mean(self.logits_real_Y) + tf.reduce_mean(self.logits_fake_Y)
        self.d_loss_X = -tf.reduce_mean(self.logits_real_X) + tf.reduce_mean(self.logits_fake_X)
        self.g_loss_Y = -tf.reduce_mean(self.logits_fake_Y) + 10. * self.L_cyc
        self.g_loss_X = -tf.reduce_mean(self.logits_fake_X) + 10. * self.L_cyc
        self.Dx_opt = tf.train.AdamOptimizer(2e-4, beta1=0., beta2=0.9).minimize(self.d_loss_X, var_list=[self.Dx.var])
        self.Dy_opt = tf.train.AdamOptimizer(2e-4, beta1=0., beta2=0.9).minimize(self.d_loss_Y, var_list=[self.Dy.var])
        self.G_opt = tf.train.AdamOptimizer(2e-4, beta1=0., beta2=0.9).minimize(self.g_loss_Y, var_list=[G.var])
        self.F_opt = tf.train.AdamOptimizer(2e-4, beta1=0., beta2=0.9).minimize(self.g_loss_X, var_list=[F.var])

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.train()


    def train(self):
        Y_path = "./Y//"
        X_path = "./X//"
        Y = os.listdir(Y_path)[:img_nums]
        X = os.listdir(X_path)[:img_nums]
        nums = Y.__len__()
        saver = tf.train.Saver()
        for epoch in range(10000):
            for i in range(int(nums / batchsize) - 1):
                X_img = np.zeros([batchsize, img_H, img_W, 3])
                Y_img = np.zeros([batchsize, img_H, img_W, 3])
                for j in np.arange(i * batchsize, i * batchsize + batchsize, 1):
                    img = misc.imresize(np.array(Image.open(X_path + X[j])), [img_H, img_W])
                    X_img[j - i * batchsize, :, :, :] = img
                    img = misc.imresize(np.array(Image.open(Y_path + Y[j])), [img_H, img_W])
                    Y_img[j - i * batchsize, :, :, :] = img
                self.sess.run(self.Dy_opt, feed_dict={self.X: X_img, self.Y: Y_img})
                self.sess.run(self.Dx_opt, feed_dict={self.X: X_img, self.Y: Y_img})
                self.sess.run(self.G_opt, feed_dict={self.X: X_img, self.Y: Y_img})
                self.sess.run(self.F_opt, feed_dict={self.X: X_img, self.Y: Y_img})
                if i % 10 == 0:
                    [d_loss_X, d_loss_Y, g_loss_Y, g_loss_X, fake_X, fake_Y, cyc_loss] = \
                        self.sess.run([self.d_loss_X, self.d_loss_Y, self.g_loss_Y, self.g_loss_X, self.fake_X, self.fake_Y, self.L_cyc], feed_dict={self.X: X_img, self.Y: Y_img})
                    print("epoch: %d, step: %d, d_loss_X: %g, d_loss_Y: %g, g_loss_X: %g, g_loss_Y: %g, cyc_loss: %g"%(epoch, i, d_loss_X, d_loss_Y, g_loss_X, g_loss_Y, cyc_loss))
                    Image.fromarray(np.uint8(fake_Y)[0, :, :, :]).save(".//fake_Y//"+str(epoch)+"_"+str(i)+".jpg")
                    Image.fromarray(np.uint8(fake_X)[0, :, :, :]).save(".//fake_X//" + str(epoch) + "_" + str(i) + ".jpg")
            saver.save(self.sess, "./save_para//CycleGAN_man_woman.ckpt")


if __name__ == "__main__":
    cyc = CycleGAN()
    pass
