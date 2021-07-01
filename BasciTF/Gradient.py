import tensorflow as tf

x = tf.Variable(initial_value=3, dtype=tf.float32)

# define y=x^2
with tf.GradientTape() as tape:
    y = tf.square(x)

y_gard = tape.gradient(y, x)
print(y, y_gard)
