import tensorflow as tf
#importing MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data",one_hot=True)

#swtting up the parameters
learning_rate=0.01
training_iteration=30
batch_size=100
display_step=2

x=tf.placeholder("float",[None,784])
y=tf.placeholder("float",[None,10])

#creating the model

#setting up the weights

W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))

with tf.name_scope("Wx_b") as scope:
	#construct a linear model
	model=tf.nn.softmax(tf.matmul(x,W)+b)

#adding summary ops to the collected data

w_h=tf.summary.histogram("weights",W)
b_h=tf.summary.histogram("biases",b)

#more name scopes will clean up the graph representation
with tf.name_scope("cost_function") as scope:
	#using cross entropy to minimize error
	#cross entropy function
	cost_function= -tf.reduce_sum(y*tf.log(model))
	#creating a summary to monitor the cost function
	tf.summary.scalar("cost_function",cost_function)

with tf.name_scope("train") as scope:
	#using gradient descent
	optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

#initializing the variables
init=tf.initialize_all_variables()	

#merge all summeries into a single operator
merged_summary_op=tf.contrib.deprecated.merge_all_summaries()

#launching the graph

with tf.Session() as sess:
	sess.run(init)

	#setting the logs writer to the folder 
	summary_writer=tf.summary.FileWriter("Desktop\Theano\100 days",graph_def=sess.graph_def)

	#training cycle

	for iteration in range(training_iteration):
		avg_cost=0
		total_batch=int(mnist.train.num_examples/batch_size)
		#loop over all batches
		for i in range(total_batch):
			batch_xs,batch_ys=mnist.train.next_batch(batch_size)
			#fit training using batch data
			sess.run(optimizer,feed_dict={x: batch_xs, y: batch_ys})
			avg_cost+=sess.run(cost_function,feed_dict={x:batch_xs, y: batch_ys})/total_batch
			#write logs for each iteration
			summary_str=sess.run(merged_summary_op,feed_dict={x: batch_xs, y: batch_ys})
			summary_writer.add_summary(summary_str,iteration*total_batch+i)

		#display logs per iteration step
		if(iteration%display_step==0):
			print("Iteration","%04d"%(iteration+1),"cost=","{:.9f}".format(avg_cost))	

	print("Tuning complete")

	#testing the model
	predictions=tf.equal(tf.argmax(model,1),tf.argmax(y,1))
	#calculate accuracy
	accuracy=tf.reduce_mean(tf.cast(predictions,"float"))
	print("Accuracy: ",accuracy.eval({x: mnist.test.images,y: mnist.test.labels}))
