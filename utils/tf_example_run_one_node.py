# Note:
# python -m venv python-env
# source python-env/bin/activate
# pip install --upgrade pip
# pip install tensorflow

import tensorflow as tf

def run_scatter_nd_update():
    ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
    expected_rsult = tf.Variable([1, 11, 3, 10, 9, 6, 12, 8])
    indices = tf.constant([[4], [3], [1] ,[7]])
    updates = tf.constant([9, 10, 11, 12])
    result = tf.compat.v1.scatter_nd_update(ref, indices, updates)
    # with tf.compat.v1.Session() as sess:
    #     print(sess.run(result))
    print("result=", result)
    if tf.math.equal(expected_rsult, result) is False:
        print("Fail: Real result != Expected result.")

def run_scatter_nd_update2():
    # data shape=[2,2,2]
    data = tf.Variable([[[1, 2], 
                        [3, 4]], 
                       [[5, 6], 
                        [7, 8]]])
    expected_rsult = tf.Variable([[[9, 10], 
                                   [11, 12]], 
                                  [[5, 6], 
                                   [7, 8]]])
    indices = tf.constant([[0]])
    updates = tf.constant([[[0, 0], [0, 0]]])
    print("data shape=", data.shape)
    print("indices shape=", indices.shape)
    print("updates shape=", updates.shape)
    result = tf.compat.v1.scatter_nd_update(data, indices, updates)

    print("result=", result)
    if tf.math.equal(expected_rsult, result) is False:
        print("Fail: Real result != Expected result.")

if __name__ == "__main__":
    # run_scatter_nd_update()
    run_scatter_nd_update2()