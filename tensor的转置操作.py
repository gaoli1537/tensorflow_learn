import tensorflow as tf 
x=tf.constant(
    [
    [1,2,3],
    [4,5,6]
    ],tf.float16)

r=tf.transpose(x,perm=[1,0])

x_3_3_4=tf.constant(
    [
        [[1,2,3,4],[5,6,7,8],[9,10,11,12]],
        [[13,14,15,16],[17,18,19,20],[21,22,23,24]],
        [[25,26,27,28],[29,30,31,32],[33,34,35,36]]
    ],tf.float16)

x_3_3_4_shape=tf.shape(x_3_3_4)

x_2_2_1=tf.reshape(x_3_3_4,[2,2,-1])
x_9_9_1=tf.reshape(x_3_3_4,[4,3,3])
session=tf.Session()
print(session.run(x_3_3_4_shape))
print(session.run(x_3_3_4))
print(session.run(x_2_2_1))
print(session.run(x_9_9_1))

print(12)

print(22)