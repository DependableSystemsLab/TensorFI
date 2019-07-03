import os
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
import driving_data
import model
import scipy.misc
import numpy as np


LOGDIR = './save'

sess = tf.InteractiveSession()

L2NormConst = 0.001

train_vars = tf.trainable_variables()

loss = tf.reduce_mean(tf.square(tf.subtract(model.y_, model.y))) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
sess.run(tf.initialize_all_variables())



#sess = tf.InteractiveSession()
#saver = tf.train.Saver()
#saver.restore(sess, "save/model.ckpt")


# create a summary to monitor cost tensor
tf.summary.scalar("loss", loss)
# merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V2)





# op to write logs to Tensorboard
logs_path = './logs'
summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

epochs = 1
batch_size = 100



xs = []
ys = []
index = [20, 486, 992, 1398, 4429, 5259, 5868, 6350, 6650, 7771]

with open("driving_dataset/data.txt") as f:
    for line in f:
        xs.append("driving_dataset/" + line.split()[0])
        #the paper by Nvidia uses the inverse of the turning radius,
        #but steering wheel angle is proportional to the inverse of turning radius
        #so the steering wheel angle in radians is used as the output
        ys.append(float(line.split()[1]) * scipy.pi / 180)

xs = xs[:10000]
ys = ys[:10000]



epochs = 10000

# train over the dataset about 30 times
for epoch in range(epochs):

  for each in index:
    full_image = scipy.misc.imread("driving_dataset/" + str(each) + ".jpg", mode="RGB")
    image = scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0

    img = image
    image = np.reshape(image, (1,66, 200, 3))
 

    yy = ys[each]

    yy = np.reshape(yy, (1,1))
    train_step.run(feed_dict={model.x: image, model.y_: yy})


   
    loss_value = loss.eval(feed_dict={model.x: image, model.y_: yy})
    print("Epoch: %d, Each: %d, Loss: %g" % (epoch, each, loss_value))



    if(epoch % 1000 ==0):
      degrees = model.y.eval(feed_dict={model.x: [img]})[0][0] * 180.0 / scipy.pi 

      print(each, yy[0][0], degrees)


      checkpoint_path = os.path.join(LOGDIR, "model.ckpt")
      filename = saver.save(sess, checkpoint_path)



    # write logs at every iteration
#    summary = merged_summary_op.eval(feed_dict={model.x: image, model.y_: ys})
#    summary_writer.add_summary(summary, epoch * driving_data.num_images/batch_size + i)


print("Model saved in file: %s" % filename)

print("Run the command line:\n" \
          "--> tensorboard --logdir=./logs " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")
