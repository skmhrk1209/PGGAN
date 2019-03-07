import tensorflow as tf


class GAN(object):

    def __init__(self, generator, discriminator, real_input_fn, fake_input_fn, hyper_params):

        # =========================================================================================
        real_images = real_input_fn()
        fake_latents = fake_input_fn()
        # =========================================================================================
        fake_images = generator(fake_latents)
        # =========================================================================================
        real_logits = discriminator(real_images)
        fake_logits = discriminator(fake_images)
        # =========================================================================================
        # WGAN-GP + ACGAN
        # [Improved Training of Wasserstein GANs]
        # (https://arxiv.org/pdf/1704.00028.pdf)
        # [Conditional Image Synthesis With Auxiliary Classifier GANs]
        # (https://arxiv.org/pdf/1610.09585.pdf)
        # -----------------------------------------------------------------------------------------
        # generator
        # wasserstein loss
        generator_losses = -fake_logits[:, 0]
        # auxiliary classification loss
        if hyper_params.generator_auxiliary_classification_weight:
            generator_auxiliary_classification_losses = tf.nn.softmax_cross_entropy_with_logits(labels=fake_labels, logits=fake_logits[:, 1:])
            generator_losses += hyper_params.generator_auxiliary_classification_weight * generator_auxiliary_classification_losses
        # -----------------------------------------------------------------------------------------
        # discriminator
        # wasserstein loss
        discriminator_losses = -real_logits[:, 0] + fake_logits[:, 0]
        # one-centered gradient penalty
        if hyper_params.one_centered_gradient_penalty_weight:
            def lerp(a, b, t): return t * a + (1. - t) * b
            coefficients = tf.random_uniform([tf.shape(real_images)[0], 1, 1, 1])
            interpolated_images = lerp(real_images, fake_images, coefficients)
            interpolated_logits = discriminator(interpolated_images)
            interpolated_gradients = tf.gradients(interpolated_logits[:, 0], [interpolated_images])[0]
            interpolated_gradient_penalties = tf.square(1. - tf.sqrt(tf.reduce_sum(tf.square(interpolated_gradients), axis=[1, 2, 3]) + 1e-8))
            discriminator_losses += hyper_params.one_centered_gradient_penalty_weight * interpolated_gradient_penalties
        # auxiliary classification loss
        if hyper_params.discriminator_auxiliary_classification_weight:
            discriminator_auxiliary_classification_losses = tf.nn.softmax_cross_entropy_with_logits(labels=real_labels, logits=real_logits[:, 1:])
            discriminator_auxiliary_classification_losses += tf.nn.softmax_cross_entropy_with_logits(labels=fake_labels, logits=fake_logits[:, 1:])
            discriminator_losses += hyper_params.discriminator_auxiliary_classification_weight * discriminator_auxiliary_classification_losses
        # =========================================================================================
        # losss reduction
        self.generator_loss = tf.reduce_mean(generator_losses)
        self.discriminator_loss = tf.reduce_mean(discriminator_losses)
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
        self.generator_train_op = generator_optimizer.minimize(
            loss=self.generator_loss,
            var_list=generator_variables,
            global_step=tf.train.get_or_create_global_step()
        )
        self.discriminator_train_op = discriminator_optimizer.minimize(
            loss=self.discriminator_loss,
            var_list=discriminator_variables
        )
        # =========================================================================================
        # scaffold
        self.scaffold = tf.train.Scaffold(
            init_op=tf.global_variables_initializer(),
            local_init_op=tf.tables_initializer(),
            saver=tf.train.Saver(
                max_to_keep=10,
                keep_checkpoint_every_n_hours=12,
            ),
            summary_op=tf.summary.merge([
                tf.summary.image(
                    name="real_images",
                    tensor=tf.transpose(real_images, [0, 2, 3, 1]),
                    max_outputs=4
                ),
                tf.summary.image(
                    name="fake_images",
                    tensor=tf.transpose(fake_images, [0, 2, 3, 1]),
                    max_outputs=4
                ),
                tf.summary.scalar(
                    name="generator_loss",
                    tensor=self.generator_loss
                ),
                tf.summary.scalar(
                    name="discriminator_loss",
                    tensor=self.discriminator_loss
                ),
            ])
        )

    def train(self, total_steps, model_dir, save_checkpoint_steps,
              save_summary_steps, log_step_count_steps, config):

        with tf.train.SingularMonitoredSession(
            scaffold=self.scaffold,
            checkpoint_dir=model_dir,
            config=config,
            hooks=[
                tf.train.CheckpointSaverHook(
                    checkpoint_dir=model_dir,
                    save_steps=save_checkpoint_steps,
                    scaffold=self.scaffold,
                ),
                tf.train.SummarySaverHook(
                    output_dir=model_dir,
                    save_steps=save_summary_steps,
                    scaffold=self.scaffold
                ),
                tf.train.LoggingTensorHook(
                    tensors=dict(
                        global_step=tf.train.get_global_step(),
                        generator_loss=self.generator_loss,
                        discriminator_loss=self.discriminator_loss
                    ),
                    every_n_iter=log_step_count_steps,
                ),
                tf.train.StepCounterHook(
                    output_dir=model_dir,
                    every_n_steps=log_step_count_steps,
                ),
                tf.train.StopAtStepHook(
                    last_step=total_steps
                )
            ]
        ) as session:

            while not session.should_stop():
                session.run(self.discriminator_train_op)
                session.run(self.generator_train_op)
