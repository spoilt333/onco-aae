import numpy as np
import tensorflow as tf

    
def get_input_data(data_file_name, unique_data_file_name):
    data = np.load(data_file_name)
    # 166 bits fingerprint, 1 concentration float, 1 TGI float
    unique_fp = np.load(unique_data_file_name)
    # there is 6252 unique fingerprints and multiple experiments with each

    np.random.shuffle(data)
    test_data, train_data = np.vsplit(data, [100])

    return test_data, train_data, unique_fp

def sample_prior(size=(64, 4)):
    return np.random.normal(size=size)

def batch_gen(data, batch_size=64):
    max_index = data.shape[0]/batch_size

    while True:
        np.random.shuffle(data)
        for i in xrange(max_index):
            yield np.hsplit(data[batch_size*i:batch_size*(i+1)], [-2, -1])

def same_gen(unq_fp, n_examples=64, n_different=1, mu=-5.82, std=1.68):
    '''
    Generator of same fingerprints with different concentraition
    '''
    
    if n_examples % n_different: 
        raise ValueError('n_examples(%s) must be divisible by n_different(%s)' % (n_examples, n_different))
    max_index = unq_fp.shape[0] / n_different
    targets = np.zeros((n_examples, n_examples))
    block_size = n_examples/n_different
    for i in xrange(n_different):
        '''blocks of ones for every block of equal fp's'''
        targets[i*block_size:(i+1)*block_size, i*block_size:(i+1)*block_size] = 1.
    targets = targets > 0
    
    while 1:
        np.random.shuffle(unq_fp)
        for i in xrange(max_index):
            batch_conc = np.random.normal(mu, std, size=(n_examples, 1))
            batch_fp = np.repeat(unq_fp[i*n_different:(i+1)*n_different], [block_size]*n_different, axis=0)
            yield batch_fp, batch_conc, targets
            
            
def uniform_initializer(size_1, size_2):
    normalized_size = np.sqrt(6) / (np.sqrt(size_1 + size_2))
    return tf.random_uniform([size_1, size_2], minval=-normalized_size, maxval=normalized_size)

def gauss_initializer(size_1, size_2):
    return tf.random_normal([size_1, size_2], 0, 2. / (size_1 * size_2))

def identity_function(x, name):
    return x

def get_collections_from_scope(scope_name):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)


def layer_output(size_1, size_2, layer_input, layer_name, activation_function=tf.nn.relu, initialization_function=uniform_initializer, batch_normed=True, epsilon = 1e-3):
    w = tf.Variable(initialization_function(size_1, size_2), name="W")
    if not batch_normed:
        b = tf.Variable(tf.random_normal([size_2]), name="b")
        return activation_function(tf.add(tf.matmul(layer_input, w), b), name=layer_name)        
    pre_output = tf.matmul(layer_input, w)
    batch_mean, batch_var = tf.nn.moments(pre_output,[0])
    scale = tf.Variable(tf.ones([size_2]))
    beta = tf.Variable(tf.zeros([size_2]))
    layer_output_value = tf.nn.batch_normalization(pre_output, batch_mean, batch_var, beta, scale, epsilon)
    return activation_function(layer_output_value, name=layer_name)


class AAE(object):
    def __init__(self, 
                 pretrain_batch_size=512, 
                 batch_size=64, 
                 latent_space=4, 
                 input_space=166, 
                 learning_rate=0.01,
                 encoder_layers=2, 
                 decoder_layers=2,
                 discriminator_layers=1,
                 initializer=gauss_initializer):

        self.pretrain_batch_size = pretrain_batch_size
        self.batch_size = batch_size
        self.latent_space = latent_space
        self.learning_rate = learning_rate
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.discriminator_layers = discriminator_layers
        self.input_space = input_space

        self.initializer = initializer

        self.fingerprint_tensor = tf.placeholder(tf.float32, [None, input_space])
        self.prior_tensor = tf.placeholder(tf.float32, [None, latent_space])
        self.conc_tensor = tf.placeholder(tf.float32, [None, 1])
        self.tgi_tensor = tf.placeholder(tf.float32, [None, 1])
        self.targets_tensor = tf.placeholder(tf.bool, [None, None])

        self.visible_tensor = tf.concat(1, [self.fingerprint_tensor, self.conc_tensor])
        self.hidden_tensor = tf.concat(1, [self.prior_tensor, self.tgi_tensor])

        # Encoder net: 166+1->128->64->3+1
        
        with tf.name_scope("encoder"):
            encoder = [self.visible_tensor]
            
            sizes = [self.input_space + 1, 128, 64, self.latent_space]

            for layer_number in xrange(encoder_layers):
                with tf.name_scope("encoder-%s" % layer_number):
                    enc_l = layer_output(sizes[layer_number], sizes[layer_number + 1], encoder[-1], 'enc_l')
                    encoder.append(enc_l)

            with tf.name_scope("encoder-fp"):
                self.encoded_fp = layer_output(sizes[-2], sizes[-1],  encoder[-1], 'encoded_fp', batch_normed=False, activation_function=identity_function)

        with tf.name_scope("tgi-encoder"):
            self.encoded_tgi = layer_output(sizes[-2], 1,  encoder[-1], 'encoded_tgi', batch_normed=False, activation_function=identity_function)

        self.encoded = tf.concat(1, [self.encoded_fp, self.encoded_tgi])
        
        # Decoder net: 3+1->64->128->166+1

        sizes = [self.latent_space + 1, 64, 128, self.input_space]

        with tf.name_scope("decoder"):
            decoder = [self.encoded]
            generator = [self.hidden_tensor]

            for layer_number in xrange(decoder_layers):
                with tf.name_scope("decoder-%s" % layer_number):
                    w = tf.Variable(self.initializer(sizes[layer_number], sizes[layer_number + 1]), name="W")
                    b = tf.Variable(tf.random_normal([sizes[layer_number + 1]]), name="b")
                    dec_l = tf.nn.tanh(tf.add(tf.matmul(decoder[-1], w), b), name="dec_l")
                    gen_l = tf.nn.tanh(tf.add(tf.matmul(generator[-1], w), b), name="gen_l")
                    decoder.append(dec_l)
                    generator.append(gen_l)

            with tf.name_scope("decoder-fp"):
                w = tf.Variable(self.initializer(sizes[-2], sizes[-1]), name="W")
                b = tf.Variable(tf.random_normal([sizes[-1]]), name="b")
                self.decoded_fp = tf.add(tf.matmul(decoder[-1], w), b, name="decoder_fp")
                self.gen_fp = tf.nn.relu(tf.add(tf.matmul(generator[-1], w), b), name="gen_fp")

            with tf.name_scope("decoder-conc"):
                w = tf.Variable(self.initializer(sizes[-2], 1), name="W")
                b = tf.Variable(tf.random_normal([1]), name="b")
                self.decoded_conc = tf.add(tf.matmul(decoder[-1], w), b)
                self.gen_conc = tf.add(tf.matmul(generator[-1], w), b)

        # Discriminator net: 3->64->3->1
        with tf.name_scope("discriminator"):
            discriminator_enc = [self.encoded_fp]
            discriminator_prior = [self.prior_tensor]

            sizes = [self.latent_space, 2 * self.latent_space - 2, 1]

            for layer_number in xrange(discriminator_layers):
                with tf.name_scope("discriminator-%s" % layer_number):
                    w = tf.Variable(self.initializer(sizes[layer_number], sizes[layer_number + 1]), name="W")
                    b = tf.Variable(tf.random_normal([sizes[layer_number + 1]]), name="b")
                    disc_enc = tf.nn.relu(tf.add(tf.matmul(discriminator_enc[-1], w), b), name="disc_enc")
                    disc_prior = tf.nn.relu(tf.add(tf.matmul(discriminator_prior[-1], w), b), name="disc_prior")
                
                    discriminator_enc.append(disc_enc)
                    discriminator_prior.append(disc_prior)

            with tf.name_scope("discriminator-final"):
                w = tf.Variable(self.initializer(sizes[-2], sizes[-1]), name="W")
                b = tf.Variable(tf.random_normal([sizes[-1]]), name="b")
                self.disc_enc = tf.add(tf.matmul(discriminator_enc[-1], w), b, name="disc_enc")
                self.disc_prior = tf.add(tf.matmul(discriminator_prior[-1], w), b, name="disc_prior")

                
        self.disc_loss = tf.reduce_mean(tf.nn.relu(self.disc_prior) - self.disc_prior + tf.log(1.0 + tf.exp(-tf.abs(self.disc_prior)))) + \
            tf.reduce_mean(tf.nn.relu(self.disc_enc) + tf.log(1.0 + tf.exp(-tf.abs(self.disc_enc))))

        fp_norms = tf.sqrt(tf.reduce_sum(tf.square(self.encoded_fp), keep_dims=True, reduction_indices=[1]))
        normalized_fp = tf.div(self.encoded_fp, fp_norms)
        cosines_fp = tf.matmul(normalized_fp, tf.transpose(normalized_fp))
        self.manifold_cost = tf.reduce_mean(1 - tf.boolean_mask(cosines_fp, self.targets_tensor))
            
        self.enc_fp_loss = tf.reduce_mean(tf.nn.relu(self.disc_enc) - self.disc_enc + tf.log(1.0 + tf.exp(-tf.abs(self.disc_enc))))
        self.enc_tgi_loss = tf.reduce_mean(tf.square(tf.sub(self.tgi_tensor, self.encoded_tgi)))
        self.enc_loss = self.enc_fp_loss

        self.dec_fp_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.decoded_fp, self.fingerprint_tensor))
        self.dec_conc_loss = tf.reduce_mean(tf.square(tf.sub(self.conc_tensor, self.decoded_conc)))
        self.dec_loss = self.dec_fp_loss + self.dec_conc_loss
        
        self.train_discriminator = tf.train.AdamOptimizer(self.learning_rate).minimize(self.disc_loss, var_list=get_collections_from_scope('discriminator'))
        self.train_encoder = tf.train.AdamOptimizer(self.learning_rate).minimize(self.enc_loss, var_list=get_collections_from_scope('encoder'))
        self.train_manifold = tf.train.AdamOptimizer(self.learning_rate).minimize(self.manifold_cost, var_list=get_collections_from_scope('encoder'))
        self.train_reg = tf.train.AdamOptimizer(self.learning_rate).minimize(self.enc_tgi_loss, var_list=get_collections_from_scope('encoder') + get_collections_from_scope('tgi-encoder'))
        self.train_autoencoder = tf.train.AdamOptimizer(self.learning_rate).minimize(self.dec_loss, var_list=get_collections_from_scope('encoder') + get_collections_from_scope('tgi-encoder') + get_collections_from_scope('decoder'))
                
        
        self.test_data, self.train_data, self.unique_fp = get_input_data("mcf7.data.npy", "mcf7.unique.fp.npy")


        
    def train(self):
        init = tf.global_variables_initializer()
        sess = tf.Session()
        saver = tf.train.Saver()

        sames = same_gen(self.unique_fp, n_different=32)

        self.pretrain(self.train_data, sess, init, sames)
        #sess.run(init)
        
        batches = batch_gen(self.train_data, self.batch_size)
        sames = same_gen(self.unique_fp, n_different=32)
        for e in xrange(50):
            if e > 0 and e % 10 == 0:
                saver.save(sess, './adam.aae.manifold.%de.model.ckpt' % e)
            
            print("epoch #%d" % e)
            
            for u in xrange(10000):
                batch_fp, batch_conc, batch_tgi = batches.next()
                batch_prior = sample_prior()
                sess.run(self.train_discriminator, feed_dict={self.fingerprint_tensor: batch_fp,
                                                         self.conc_tensor: batch_conc,
                                                         self.tgi_tensor: batch_tgi,
                                                         self.prior_tensor: batch_prior})

                batch_fp, batch_conc, batch_tgi = batches.next()

                sess.run(self.train_encoder, feed_dict={self.fingerprint_tensor: batch_fp,
                                                   self.conc_tensor: batch_conc})

        #         same_fp, same_conc, targets = sames.next()
        #         sess.run(train_manifold, feed_dict={fingerprint_tensor: batch_fp,
        #                                             conc_tensor: batch_conc,
        #                                             targets_tensor: targets})
            
                batch_fp, batch_conc, batch_tgi = batches.next()
                sess.run(self.train_reg, feed_dict={self.fingerprint_tensor: batch_fp,
                                               self.conc_tensor: batch_conc,
                                               self.tgi_tensor: batch_tgi})

                batch_fp, batch_conc, batch_tgi = batches.next()
                sess.run(self.train_autoencoder, feed_dict={self.fingerprint_tensor: batch_fp,
                                                       self.conc_tensor: batch_conc,
                                                       self.tgi_tensor: batch_tgi})
                
            else:
                batch_prior = sample_prior((100, self.latent_space))
                losses = sess.run([self.disc_loss, self.enc_fp_loss, self.enc_tgi_loss, self.dec_fp_loss, self.dec_conc_loss],
                                  feed_dict={self.fingerprint_tensor: self.train_data[:, :-2],
                                             self.conc_tensor: self.train_data[:, -2:-1],
                                             self.tgi_tensor: self.train_data[:, -1:],
                                             self.prior_tensor: batch_prior
                                            })
                
                same_fp, same_conc, targets = sames.next()
                m_loss = sess.run(self.manifold_cost, feed_dict={self.fingerprint_tensor: batch_fp,
                                                            self.conc_tensor: batch_conc,
                                                            self.targets_tensor: targets})
                
                discriminator_loss, encoder_fp_loss, encoder_tgi_loss, autoencoder_fp_loss, autoencoder_conc_loss = losses
                print("disc: %f, enc_fp : %f, mani_fp: %f, enc_tgi: %f, dec_fp : %f, dec_conc : %f" % (discriminator_loss/2.,
                                                                                                       encoder_fp_loss,
                                                                                                       m_loss,
                                                                                                       encoder_tgi_loss,
                                                                                                       autoencoder_fp_loss,
                                                                                                       autoencoder_conc_loss))

    def pretrain(self, train_data, sess, init, sames):
        # pretrain generator w/o regressions and decoding

        batches = batch_gen(train_data, self.pretrain_batch_size)
        
        flag = True
        while flag:
            # need to do a few initialization tries, because from some points
            # Generator doesn't converge.
            
            sess.run(init)
            for e in xrange(15):
                print("epoch #%d" % e)
                discriminator_loss = 0.0
                encoder_fp_loss = 0.0
                mani_loss = 0.0
                for u in xrange(1000):
                    batch_fp, batch_conc, _ = batches.next()
                    batch_prior = sample_prior()
                    _, loss = sess.run([self.train_discriminator, self.disc_loss], 
                        feed_dict={self.fingerprint_tensor: batch_fp,
                            self.conc_tensor: batch_conc,
                            self.prior_tensor: batch_prior})
                    discriminator_loss += loss

                    fp_loss = 2.
                    count = 0
                    while fp_loss > 1. and count < 20:
                        batch_fp, batch_conc, _ = batches.next()
                        _, fp_loss = sess.run([self.train_encoder, self.enc_fp_loss], 
                                            feed_dict={self.fingerprint_tensor: batch_fp,
                                            self.conc_tensor: batch_conc,})
                        count += 1
                    else:
                        encoder_fp_loss += fp_loss

                    same_fp, same_conc, targets = sames.next()
                    _, m_loss = sess.run([self.train_manifold, self.manifold_cost], 
                        feed_dict={self.fingerprint_tensor: batch_fp,
                        self.conc_tensor: batch_conc,
                        self.targets_tensor: targets})
                    
                    mani_loss += m_loss

                discriminator_loss /= 1000. * 2.
                encoder_fp_loss /= 1000.
                mani_loss /= 1000.

                print("disc: %f, enc_p: %f, mani: %f" % (discriminator_loss, encoder_fp_loss, mani_loss))
                if (e >= 5) and (encoder_fp_loss < 0.7):
                    flag = False
                    break


