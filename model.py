import tensorflow as tf
import numpy as np
import metrics


class GAN(object):

    def __init__(self, generator, discriminator, real_input_fn, fake_input_fn, hyper_params):

        # =========================================================================================
        real_images, labels = real_input_fn()
        fake_latents = fake_input_fn()
        # =========================================================================================
        fake_images = generator(fake_latents)
        # =========================================================================================
        real_adversarial_logits, real_classification_logits = discriminator(real_images, labels)
        fake_adversarial_logits, fake_classification_logits = discriminator(fake_images, labels)
        # =========================================================================================
        # WGAN-GP + ACGAN
        # [Improved Training of Wasserstein GANs]
        # (https://arxiv.org/pdf/1704.00028.pdf)
        # [Conditional Image Synthesis With Auxiliary Classifier GANs]
        # (https://arxiv.org/pdf/1610.09585.pdf)
        # -----------------------------------------------------------------------------------------
        # wasserstein loss
        generator_losses = -fake_adversarial_logits
        # auxiliary classification loss
        if hyper_params.generator_classification_loss_weight:
            generator_classification_losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_classification_logits)
            generator_losses += generator_classification_losses * hyper_params.generator_classification_loss_weight
        # -----------------------------------------------------------------------------------------
        # wasserstein loss
        discriminator_losses = -real_adversarial_logits
        discriminator_losses += fake_adversarial_logits
        # one-centered gradient penalty
        if hyper_params.gradient_penalty_weight:
            def lerp(a, b, t): return t * a + (1 - t) * b
            coefficients = tf.random_uniform([tf.shape(real_images)[0], 1, 1, 1])
            interpolated_images = lerp(real_images, fake_images, coefficients)
            interpolated_adversarial_logits, _ = discriminator(interpolated_images, labels)
            interpolated_gradients = tf.gradients(interpolated_adversarial_logits, [interpolated_images])[0]
            interpolated_gradient_penalties = tf.square(1 - tf.sqrt(tf.reduce_sum(tf.square(interpolated_gradients), axis=[1, 2, 3]) + 1e-8))
            discriminator_losses += interpolated_gradient_penalties * hyper_params.gradient_penalty_weight
        # auxiliary classification loss
        if hyper_params.discriminator_classification_loss_weight:
            discriminator_classification_losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=real_classification_logits)
            discriminator_classification_losses += tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_classification_logits)
            discriminator_losses += discriminator_classification_losses * hyper_params.discriminator_classification_loss_weight
        # =========================================================================================
        # losss reduction
        generator_loss = tf.reduce_mean(generator_losses)
        discriminator_loss = tf.reduce_mean(discriminator_losses)
        # =========================================================================================
        generator_optimizer = tf.train.AdamOptimizer(
            learning_rate=hyper_params.generator_learning_rate,
            beta1=hyper_params.generator_beta1,
            beta2=hyper_params.generator_beta2
        )
        discriminator_optimizer = tf.train.AdamOptimizer(
            learning_rate=hyper_params.discriminator_learning_rate,
            beta1=hyper_params.discriminator_beta1,
            beta2=hyper_params.discriminator_beta2
        )
        # -----------------------------------------------------------------------------------------
        generator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
        discriminator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
        # =========================================================================================
        generator_train_op = generator_optimizer.minimize(
            loss=generator_loss,
            var_list=generator_variables,
            global_step=tf.train.get_or_create_global_step()
        )
        discriminator_train_op = discriminator_optimizer.minimize(
            loss=discriminator_loss,
            var_list=discriminator_variables
        )
        # =========================================================================================
        self.real_images = tf.transpose(real_images, [0, 2, 3, 1])
        self.fake_images = tf.transpose(fake_images, [0, 2, 3, 1])
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss
        self.generator_train_op = generator_train_op
        self.discriminator_train_op = discriminator_train_op

    def train(self, model_dir, total_steps, save_checkpoint_steps, save_summary_steps, log_tensor_steps, config):

        with tf.train.SingularMonitoredSession(
            scaffold=tf.train.Scaffold(
                init_op=tf.global_variables_initializer(),
                local_init_op=tf.group(
                    tf.local_variables_initializer(),
                    tf.tables_initializer()
                )
            ),
            checkpoint_dir=model_dir,
            config=config,
            hooks=[
                tf.train.CheckpointSaverHook(
                    checkpoint_dir=model_dir,
                    save_steps=save_checkpoint_steps,
                    saver=tf.train.Saver(
                        max_to_keep=10,
                        keep_checkpoint_every_n_hours=12,
                    )
                ),
                tf.train.SummarySaverHook(
                    output_dir=model_dir,
                    save_steps=save_summary_steps,
                    summary_op=tf.summary.merge(list(map(
                        lambda name_tensor: tf.summary.image(*name_tensor), dict(
                            real_images=self.real_images,
                            fake_images=self.fake_images
                        ).items()
                    )))
                ),
                tf.train.SummarySaverHook(
                    output_dir=model_dir,
                    save_steps=save_summary_steps,
                    summary_op=tf.summary.merge(list(map(
                        lambda name_tensor: tf.summary.scalar(*name_tensor), dict(
                            generator_loss=self.generator_loss,
                            discriminator_loss=self.discriminator_loss
                        ).items()
                    )))
                ),
                tf.train.LoggingTensorHook(
                    tensors=dict(
                        global_step=tf.train.get_global_step(),
                        generator_loss=self.generator_loss,
                        discriminator_loss=self.discriminator_loss
                    ),
                    every_n_iter=log_tensor_steps,
                ),
                tf.train.StopAtStepHook(
                    last_step=total_steps
                )
            ]
        ) as session:

            while not session.should_stop():
                session.run(self.discriminator_train_op)
                session.run(self.generator_train_op)

    def evaluate(self, model_dir, config):

        real_features = tf.contrib.gan.eval.run_inception(
            images=tf.contrib.gan.eval.preprocess_image(self.real_images),
            output_tensor="pool_3:0"
        )
        fake_features = tf.contrib.gan.eval.run_inception(
            images=tf.contrib.gan.eval.preprocess_image(self.fake_images),
            output_tensor="pool_3:0"
        )

        with tf.train.SingularMonitoredSession(
            scaffold=tf.train.Scaffold(
                init_op=tf.global_variables_initializer(),
                local_init_op=tf.group(
                    tf.local_variables_initializer(),
                    tf.tables_initializer()
                )
            ),
            checkpoint_dir=model_dir,
            config=config
        ) as session:

            def generator():
                while True:
                    try:
                        yield session.run([self.real_features, self.fake_features])
                    except tf.errors.OutOfRangeError:
                        break

            frechet_inception_distance = metrics.frechet_inception_distance(*map(np.concatenate, zip(*generator())))
            tf.logging.info("frechet_inception_distance: {}".format(frechet_inception_distance))
