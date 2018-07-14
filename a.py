import tensorflow as tf

def fnx(coef, x):
  return coef[0] + coef[1]*x + coef[2]*x**2 + coef[3]*x**3 + coef[4]*x**4

def fnx_d(coef, x):
  return coef[1] + 2*coef[2]*x + 3*coef[3]*x**2 + 4*coef[4]*x**3

def fnx_d2(coef, x):
  return 2*coef[2] + 6*coef[3] + 12*coef[4]*x**2 

def halley(coef, x):
  return 2*fnx(coef, x)*fnx_d(coef, x) / (2*fnx_d(coef, x)**2 - fnx(coef, x)*fnx_d2(coef, x))

with tf.Session() as sess:
  xn = tf.placeholder(dtype=tf.float32, shape=())
  coef = tf.placeholder(dtype=tf.float32, shape=(5,))
  xn1 = halley(coef, xn)
  c = lambda xn1, xn: tf.less(0.001, tf.abs(xn1-xn))
  h = tf.while_loop(c, lambda xn1, xn: [halley(coef, xn1), xn1], [xn1, xn])
  result = sess.run(h, feed_dict = {
    xn: 0.7,
    coef: [3.0, 1.5, 4.7, -0.3, 6.7],
  })
  print(result)
