import tensorflow as tf


class Net(tf.keras.Model):
    """A simple linear model."""

    def __init__(self):
        super(Net, self).__init__()
        self.l1 = tf.keras.layers.Dense(5)

    def call(self, x):
        return self.l1(x)


net = Net()


def toy_dataset():
    inputs = tf.range(10.)[:, None]
    labels = inputs * 5. + tf.range(5.)[None, :]
    return tf.data.Dataset.from_tensor_slices(
        dict(x=inputs, y=labels)).repeat().batch(2)


def train_step(net, example, optimizer):
    """Trains `net` on `example` using `optimizer`."""
    with tf.GradientTape() as tape:
        output = net(example['x'])
        loss = tf.reduce_mean(tf.abs(output - example['y']))
    variables = net.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss


lr = tf.keras.experimental.LinearCosineDecay(initial_learning_rate=0.001,
                                             decay_steps=1000,
                                             num_periods=2,
                                             alpha=0.0,
                                             beta=0.001)

opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
dataset = toy_dataset()
iterator = iter(dataset)
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=net)
manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)


def train_and_checkpoint(net, manager):
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    for _ in range(50):
        example = next(iterator)
        loss = train_step(net, example, opt)
        ckpt.step.assign_add(1)
        if int(ckpt.step) % 10 == 0:
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
            print("loss {:1.2f}".format(loss.numpy()))


train_and_checkpoint(net, manager)

pretrained_weights = net.trainable_weights
#### restore

print(opt._decayed_lr(tf.float32))
lr = tf.keras.experimental.LinearCosineDecay(initial_learning_rate=0.001,
                                             decay_steps=1000,
                                             num_periods=2,
                                             alpha=0.0,
                                             beta=0.001)

opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)

net = Net()
dataset = toy_dataset()
iterator = iter(dataset)
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=net, iterator=iterator)
manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)

train_and_checkpoint(net, manager)

print(manager.checkpoints)  # List the three remaining checkpoints

to_restore = tf.Variable(tf.zeros([5]))
print(to_restore.numpy())  # All zeros
fake_layer = tf.train.Checkpoint(bias=to_restore)
fake_net = tf.train.Checkpoint(l1=fake_layer)
new_root = tf.train.Checkpoint(net=fake_net)
status = new_root.restore(tf.train.latest_checkpoint('./tf_ckpts/'))
print(to_restore.numpy())  # This gets the restored value.
