from __future__ import division
from __future__ import print_function

import prettytensor as pt
import tensorflow as tf
import numpy as np
import scipy.misc
import os
import sys
from six.moves import range
from progressbar import ETA, Bar, Percentage, ProgressBar
import functools


from misc.config import cfg
from misc.utils import mkdir_p

TINY = 1e-8

tf.flags.DEFINE_integer('batch_size', 64, 'batch size')

tf.flags.DEFINE_integer('crop_size', 64, 'crop size')

tf.flags.DEFINE_string('data_dir', 'Data/birds/CUB_200_2011/', 'data')

tf.flags.DEFINE_integer('sample_num', 4, 'context sample num')

FLAGS = tf.flags.FLAGS


# reduce_mean normalize also the dimension of the embeddings
def KL_loss(mu, log_sigma):
    with tf.name_scope("KL_divergence"):
        loss = -log_sigma + .5 * (-1 + tf.exp(2. * log_sigma) + tf.square(mu))
        loss = tf.reduce_mean(loss)
        return loss


class CondGANTrainer(object):
    def __init__(self,
                 model,
                 dataset=None,
                 exp_name="model",
                 ckt_logs_dir="ckt_logs",
                 ):
        """
        :type model: RegularizedGAN
        """
        self.model = model
        self.dataset = dataset
        self.exp_name = exp_name
        self.log_dir = ckt_logs_dir
        self.checkpoint_dir = ckt_logs_dir

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
        self.model_path = cfg.TRAIN.PRETRAINED_MODEL

        self.log_vars = []

    def build_placeholder(self):
        '''Helper function for init_opt'''
        self.images = tf.placeholder(
            tf.float32, [self.batch_size] + self.dataset.image_shape,
            name='real_images')
        self.wrong_images = tf.placeholder(
            tf.float32, [self.batch_size] + self.dataset.image_shape,
            name='wrong_images'
        )
        self.embeddings = tf.placeholder(
            tf.float32, [self.batch_size] + self.dataset.embedding_shape,
            name='conditional_embeddings'
        )

        self.generator_lr = tf.placeholder(
            tf.float32, [],
            name='generator_learning_rate'
        )
        self.discriminator_lr = tf.placeholder(
            tf.float32, [],
            name='discriminator_learning_rate'
        )

    def sample_encoded_context(self, embeddings):
        '''Helper function for init_opt'''
        c_mean_logsigma = self.model.generate_condition(embeddings)
        mean = c_mean_logsigma[0]
        if cfg.TRAIN.COND_AUGMENTATION:
            # epsilon = tf.random_normal(tf.shape(mean))
            epsilon = tf.truncated_normal(tf.shape(mean))
            stddev = tf.exp(c_mean_logsigma[1])
            c = mean + stddev * epsilon

            kl_loss = KL_loss(c_mean_logsigma[0], c_mean_logsigma[1])
        else:
            c = mean
            kl_loss = 0

        return c, cfg.TRAIN.COEFF.KL * kl_loss

    def init_opt(self):
        self.build_placeholder()

        with pt.defaults_scope(phase=pt.Phase.train):
            with tf.variable_scope("g_net"):
                # ####get output from G network################################
                c, kl_loss = self.sample_encoded_context(self.embeddings)
                z = tf.random_normal([self.batch_size, cfg.Z_DIM])
                self.log_vars.append(("hist_c", c))
                self.log_vars.append(("hist_z", z))
                fake_images = self.model.get_generator(tf.concat([c, z], axis=1))

            # ####get discriminator_loss and generator_loss ###################
            discriminator_loss, generator_loss =\
                self.compute_losses(self.images,
                                    self.wrong_images,
                                    fake_images,
                                    self.embeddings)
            generator_loss += kl_loss
            self.log_vars.append(("g_loss_kl_loss", kl_loss))
            self.log_vars.append(("g_loss", generator_loss))
            self.log_vars.append(("d_loss", discriminator_loss))

            # #######Total loss for build optimizers###########################
            self.prepare_trainer(generator_loss, discriminator_loss)
            # #######define self.g_sum, self.d_sum,....########################
            self.define_summaries()

        with pt.defaults_scope(phase=pt.Phase.test):
            with tf.variable_scope("g_net", reuse=True):
                self.sampler()
            self.visualization(cfg.TRAIN.NUM_COPY)
            print("success")

    def sampler(self):
        c, _ = self.sample_encoded_context(self.embeddings)
        if cfg.TRAIN.FLAG:
            z = tf.zeros([self.batch_size, cfg.Z_DIM])  # Expect similar BGs
        else:
            z = tf.random_normal([self.batch_size, cfg.Z_DIM])
        self.fake_images = self.model.get_generator(tf.concat([c, z], axis=1))

    def compute_losses(self, images, wrong_images, fake_images, embeddings):
        with tf.variable_scope('discriminator') as vs:
            real_logit = self.model.get_discriminator(images, embeddings)
            vs.reuse_variables()
            wrong_logit = self.model.get_discriminator(wrong_images, embeddings)
            fake_logit = self.model.get_discriminator(fake_images, embeddings)

        real_d_loss =\
            tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logit,
                                                    labels=tf.ones_like(real_logit))
        real_d_loss = tf.reduce_mean(real_d_loss)
        wrong_d_loss =\
            tf.nn.sigmoid_cross_entropy_with_logits(logits=wrong_logit,
                                                    labels=tf.zeros_like(wrong_logit))
        wrong_d_loss = tf.reduce_mean(wrong_d_loss)
        fake_d_loss =\
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit,
                                                    labels=tf.zeros_like(fake_logit))
        fake_d_loss = tf.reduce_mean(fake_d_loss)
        if cfg.TRAIN.B_WRONG:
            discriminator_loss =\
                real_d_loss + (wrong_d_loss + fake_d_loss) / 2.
            self.log_vars.append(("d_loss_wrong", wrong_d_loss))
        else:
            discriminator_loss = real_d_loss + fake_d_loss
        self.log_vars.append(("d_loss_real", real_d_loss))
        self.log_vars.append(("d_loss_fake", fake_d_loss))

        generator_loss = \
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit,
                                                    labels=tf.ones_like(fake_logit))
        generator_loss = tf.reduce_mean(generator_loss)

        return discriminator_loss, generator_loss

    def prepare_trainer(self, generator_loss, discriminator_loss):
        '''Helper function for init_opt'''
        all_vars = tf.trainable_variables()

        g_vars = [var for var in all_vars if
                  var.name.startswith('g_')]
        d_vars = [var for var in all_vars if
                  var.name.startswith('dis')]

        generator_opt = tf.train.AdamOptimizer(self.generator_lr,
                                               beta1=0.5)

        gen_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'g_net')
        grads = generator_opt.compute_gradients(generator_loss, g_vars)
        with tf.control_dependencies(gen_update_ops):
            self.generator_trainer = generator_opt.apply_gradients(grads)
        discriminator_opt = tf.train.AdamOptimizer(self.discriminator_lr,
                                                   beta1=0.5)

        dis_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator')
        grads = generator_opt.compute_gradients(discriminator_loss, d_vars)
        with tf.control_dependencies(dis_update_ops):
            self.discriminator_trainer = discriminator_opt.apply_gradients(grads)

        self.log_vars.append(("g_learning_rate", self.generator_lr))
        self.log_vars.append(("d_learning_rate", self.discriminator_lr))

    def define_summaries(self):
        '''Helper function for init_opt'''
        all_sum = {'g': [], 'd': [], 'hist': []}
        for k, v in self.log_vars:
            if k.startswith('g'):
                all_sum['g'].append(tf.summary.scalar(k, v))
            elif k.startswith('d'):
                all_sum['d'].append(tf.summary.scalar(k, v))
            elif k.startswith('hist'):
                all_sum['hist'].append(tf.summary.histogram(k, v))

        self.g_sum = tf.summary.merge(all_sum['g'])
        self.d_sum = tf.summary.merge(all_sum['d'])
        self.hist_sum = tf.summary.merge(all_sum['hist'])

        all_sum = [tf.summary.histogram(variable.op.name, variable) for variable in tf.global_variables()]
        self.all_sum = tf.summary.merge(all_sum)

    def visualize_one_superimage(self, img_var, images, rows, filename):
        stacked_img = []
        for row in range(rows):
            img = images[row * rows, :, :, :]
            row_img = [img]  # real image
            for col in range(rows):
                row_img.append(img_var[row * rows + col, :, :, :])
            # each rows is 1realimage +10_fakeimage
            stacked_img.append(tf.concat(row_img, axis=1))
        imgs = tf.expand_dims(tf.concat(stacked_img, axis=0), 0)
        current_img_summary = tf.summary.image(filename, imgs)
        return current_img_summary, imgs

    def visualization(self, n):
        fake_sum_train, superimage_train = \
            self.visualize_one_superimage(self.fake_images[:n * n],
                                          self.images[:n * n],
                                          n, "train")
        fake_sum_test, superimage_test = \
            self.visualize_one_superimage(self.fake_images[n * n:2 * n * n],
                                          self.images[n * n:2 * n * n],
                                          n, "test")
        self.superimages = tf.concat([superimage_train, superimage_test], axis=0)
        self.image_summary = tf.summary.merge([fake_sum_train, fake_sum_test])

    def preprocess(self, x, n):
        # make sure every row with n column have the same embeddings
        for i in range(n):
            for j in range(1, n):
                x[i * n + j] = x[i * n]
        return x

    def epoch_sum_images(self, sess, n):
        images_train, _, embeddings_train, captions_train, _ =\
            self.dataset.train.next_batch(n * n, cfg.TRAIN.NUM_EMBEDDING)
        images_train = self.preprocess(images_train, n)
        embeddings_train = self.preprocess(embeddings_train, n)

        images_test, _, embeddings_test, captions_test, _ = \
            self.dataset.test.next_batch(n * n, 1)
        images_test = self.preprocess(images_test, n)
        embeddings_test = self.preprocess(embeddings_test, n)

        images = np.concatenate([images_train, images_test], axis=0)
        embeddings =\
            np.concatenate([embeddings_train, embeddings_test], axis=0)

        if self.batch_size > 2 * n * n:
            images_pad, _, embeddings_pad, _, _ =\
                self.dataset.test.next_batch(self.batch_size - 2 * n * n, 1)
            images = np.concatenate([images, images_pad], axis=0)
            embeddings = np.concatenate([embeddings, embeddings_pad], axis=0)
        feed_dict = {self.images: images,
                     self.embeddings: embeddings}
        gen_samples, img_summary =\
            sess.run([self.superimages, self.image_summary], feed_dict)

        # save images generated for train and test captions
        scipy.misc.imsave('%s/train.jpg' % (self.log_dir), gen_samples[0])
        scipy.misc.imsave('%s/test.jpg' % (self.log_dir), gen_samples[1])

        # pfi_train = open(self.log_dir + "/train.txt", "w")
        pfi_test = open(self.log_dir + "/test.txt", "w")
        for row in range(n):
            # pfi_train.write('\n***row %d***\n' % row)
            # pfi_train.write(captions_train[row * n])

            pfi_test.write('\n***row %d***\n' % row)
            pfi_test.write(captions_test[row * n])
        # pfi_train.close()
        pfi_test.close()

        return img_summary

    def build_model(self, sess):
        self.init_opt()
        sess.run(tf.initialize_all_variables())

        if len(self.model_path) > 0:
            print("Reading model parameters from %s" % self.model_path)
            restore_vars = tf.all_variables()
            # all_vars = tf.all_variables()
            # restore_vars = [var for var in all_vars if
            #                 var.name.startswith('g_') or
            #                 var.name.startswith('d_')]
            saver = tf.train.Saver(restore_vars)
            saver.restore(sess, self.model_path)

            istart = self.model_path.rfind('_') + 1
            iend = self.model_path.rfind('.')
            counter = self.model_path[istart:iend]
            counter = int(counter)
        else:
            print("Created model with fresh parameters.")
            counter = 0
        return counter

    def train(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        op1, op2, op3, _ = train_input_fn(FLAGS.data_dir)
        with tf.Session(config=config) as sess:
            with tf.device("/gpu:%d" % cfg.GPU_ID):
                counter = self.build_model(sess)
                saver = tf.train.Saver(tf.all_variables(),
                                       keep_checkpoint_every_n_hours=2)

                # summary_op = tf.merge_all_summaries()
                summary_writer = tf.summary.FileWriter(self.log_dir,
                                                        sess.graph)

                keys = ["d_loss", "g_loss"]
                log_vars = []
                log_keys = []
                for k, v in self.log_vars:
                    if k in keys:
                        log_vars.append(v)
                        log_keys.append(k)
                        # print(k, v)
                generator_lr = cfg.TRAIN.GENERATOR_LR
                discriminator_lr = cfg.TRAIN.DISCRIMINATOR_LR
                num_embedding = cfg.TRAIN.NUM_EMBEDDING
                lr_decay_step = cfg.TRAIN.LR_DECAY_EPOCH
                number_example = self.dataset.train._num_examples
                updates_per_epoch = int(number_example / self.batch_size)
                epoch_start = int(counter / updates_per_epoch)
                for epoch in range(epoch_start, self.max_epoch):
                    widgets = ["epoch #%d|" % epoch,
                               Percentage(), Bar(), ETA()]
                    pbar = ProgressBar(maxval=updates_per_epoch,
                                       widgets=widgets)
                    pbar.start()

                    if epoch % lr_decay_step == 0 and epoch != 0:
                        generator_lr *= 0.5
                        discriminator_lr *= 0.5

                    all_log_vals = []
                    for i in range(updates_per_epoch):
                        pbar.update(i)
                        # training d
                        # images, wrong_images, embeddings, _, _ =\
                        #     self.dataset.train.next_batch(self.batch_size,
                        #                                   num_embedding)
                        images, wrong_images, embeddings = sess.run([op1, op2, op3])
                        feed_dict = {self.images: images,
                                     self.wrong_images: wrong_images,
                                     self.embeddings: embeddings,
                                     self.generator_lr: generator_lr,
                                     self.discriminator_lr: discriminator_lr
                                     }
                        # train d
                        if i % 100 == 0 and i != 0:
                            feed_out = [self.discriminator_trainer,
                                        self.d_sum,
                                        self.hist_sum,
                                        log_vars]
                            _, d_sum, hist_sum, log_vals = sess.run(feed_out,
                                                                    feed_dict)
                            summary_writer.add_summary(d_sum, counter)
                            summary_writer.add_summary(hist_sum, counter)
                            all_log_vals.append(log_vals)
                        else:
                            sess.run(self.discriminator_trainer, feed_dict)
                        # train g
                        if i % 100 == 0 and i != 0:
                            feed_out = [self.generator_trainer,
                                        self.g_sum, self.all_sum]
                            _, g_sum, all_sum = sess.run(feed_out,
                                                feed_dict)
                            summary_writer.add_summary(g_sum, counter)
                            summary_writer.add_summary(all_sum, counter)
                        else:
                            sess.run(self.generator_trainer, feed_dict)
                        # save checkpoint
                        counter += 1
                        if counter % self.snapshot_interval == 0:
                            snapshot_path = "%s/%s_%s.ckpt" %\
                                             (self.checkpoint_dir,
                                              self.exp_name,
                                              str(counter))
                            fn = saver.save(sess, snapshot_path)
                            print("Model saved in file: %s" % fn)

                    img_sum = self.epoch_sum_images(sess, cfg.TRAIN.NUM_COPY)
                    summary_writer.add_summary(img_sum, counter)

                    avg_log_vals = np.mean(np.array(all_log_vals), axis=0)
                    dic_logs = {}
                    for k, v in zip(log_keys, avg_log_vals):
                        dic_logs[k] = v
                        # print(k, v)

                    log_line = "; ".join("%s: %s" %
                                         (str(k), str(dic_logs[k]))
                                         for k in dic_logs)
                    print("Epoch %d | " % (epoch) + log_line)
                    sys.stdout.flush()
                    if np.any(np.isnan(avg_log_vals)):
                        raise ValueError("NaN detected!")

    def save_super_images(self, images, sample_batchs, filenames,
                          sentenceID, save_dir, subset):
        # batch_size samples for each embedding
        numSamples = len(sample_batchs)
        for j in range(len(filenames)):
            s_tmp = '%s-1real-%dsamples/%s/%s' %\
                (save_dir, numSamples, subset, filenames[j])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)
            superimage = [images[j]]
            # cfg.TRAIN.NUM_COPY samples for each text embedding/sentence
            for i in range(len(sample_batchs)):
                superimage.append(sample_batchs[i][j])

            superimage = np.concatenate(superimage, axis=1)
            fullpath = '%s_sentence%d.jpg' % (s_tmp, sentenceID)
            scipy.misc.imsave(fullpath, superimage)

    def eval_one_dataset(self, sess, dataset, save_dir, subset='train'):
        count = 0
        print('num_examples:', dataset._num_examples)
        while count < dataset._num_examples:
            start = count % dataset._num_examples
            images, embeddings_batchs, filenames, _ =\
                dataset.next_batch_test(self.batch_size, start, 1)
            print('count = ', count, 'start = ', start)
            for i in range(len(embeddings_batchs)):
                samples_batchs = []
                # Generate up to 16 images for each sentence,
                # with randomness from noise z and conditioning augmentation.
                for j in range(np.minimum(16, cfg.TRAIN.NUM_COPY)):
                    samples = sess.run(self.fake_images,
                                       {self.embeddings: embeddings_batchs[i]})
                    samples_batchs.append(samples)
                self.save_super_images(images, samples_batchs,
                                       filenames, i, save_dir,
                                       subset)

            count += self.batch_size

    def evaluate(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            with tf.device("/gpu:%d" % cfg.GPU_ID):
                if self.model_path.find('.ckpt') != -1:
                    self.init_opt()
                    print("Reading model parameters from %s" % self.model_path)
                    saver = tf.train.Saver(tf.all_variables())
                    saver.restore(sess, self.model_path)
                    # self.eval_one_dataset(sess, self.dataset.train,
                    #                       self.log_dir, subset='train')
                    self.eval_one_dataset(sess, self.dataset.test,
                                          self.log_dir, subset='test')
                else:
                    print("Input a valid model path.")


def split(line):
    sp = tf.string_split([line], delimiter=',')
    name = sp.values[0]
    label = sp.values[1]
    label = tf.string_to_number(label, out_type=tf.int32)
    return name, label


def read_img(t1, t2, data_dir):
    def imread(im_path):
        img = tf.read_file(tf.string_join([data_dir, 'lr_imgs/', im_path, '.png']))
        img = tf.image.decode_png(img, 3)
        img = tf.random_crop(img, [FLAGS.crop_size, FLAGS.crop_size, 3])
        img = tf.image.random_flip_left_right(img)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = img * 2 - 1
        return img
    correct = imread(t1[0])
    wrong = imread(t2[0])
    return correct, wrong, t1[0], t1[1]


def read_sen(correct, wrong, n1, l1, data_dir):
    context = np.load(data_dir + 'sentence/' + n1 + '.npy')
    idx = np.random.choice(context.shape[0], FLAGS.sample_num, replace=False)
    context = np.mean(context[idx], axis=0)
    return correct, wrong, context, l1


def resize_img(correct, wrong, context, label):
    correct.set_shape([FLAGS.batch_size, FLAGS.crop_size, FLAGS.crop_size, 3])
    wrong.set_shape([FLAGS.batch_size, FLAGS.crop_size, FLAGS.crop_size, 3])
    context.set_shape([FLAGS.batch_size, 1024])
    label.set_shape(FLAGS.batch_size)
    return correct, wrong, context, label


def train_input_fn(data_dir, subset='train'):
    dataset = tf.data.TextLineDataset(tf.string_join([data_dir, 'my', subset, '.txt']))
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2000)
    dataset = dataset.map(split)
    dataset = tf.data.Dataset.zip((dataset, dataset))
    dataset = dataset.filter(lambda t1, t2: tf.not_equal(t1[1], t2[1]))
    dataset = dataset.map(functools.partial(read_img, data_dir=data_dir), num_parallel_calls=8)
    dataset = dataset.map(lambda c, w, n, l: tuple(tf.py_func(read_sen, [c, w, n, l, data_dir], [tf.float32, tf.float32, tf.float32, tf.int32])),
                          num_parallel_calls=8)
    dataset = dataset.batch(FLAGS.batch_size)
    dataset = dataset.map(resize_img)
    dataset = dataset.prefetch(4)
    iterator = dataset.make_one_shot_iterator()
    correct, wrong, context, targets = iterator.get_next()
    return correct, wrong, context, targets
