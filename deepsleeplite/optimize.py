import tensorflow as tf


def adam(loss, lr, train_vars, beta1=0.9, beta2=0.999, epsilon=1e-8):
    opt = tf.compat.v1.train.AdamOptimizer(
        learning_rate=lr,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        name="Adam"
    )
    grads_and_vars = opt.compute_gradients(loss, train_vars)
    apply_gradient_op = opt.apply_gradients(grads_and_vars)
    return apply_gradient_op, grads_and_vars