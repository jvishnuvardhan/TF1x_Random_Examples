## Clipping the gradients to avoid gradient exploding
```python
import tensorflow as tf
a = tf.Variable([[23, 78],
 [24, 79],
 [25, 78],
 [23, 81],
 [27, 82],
 [21, 87],
 [28, 88],
 [23, 90]])

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    clipped_value = tf.clip_by_value(a[:,1], 80, 85)
    sess.run(tf.assign(a[:,1], clipped_value))
    print(sess.run(a))
```
