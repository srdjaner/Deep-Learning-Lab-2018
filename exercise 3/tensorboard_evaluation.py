import tensorflow as tf

class Evaluation:

    def __init__(self, store_dir):
        tf.reset_default_graph()

        #self.sess = session
        self.sess = tf.Session()
        self.tf_writer = tf.summary.FileWriter(store_dir, self.sess.graph)

        #with self.sess.graph.as_default():
        self.tf_train = tf.placeholder(tf.float32, name="train_accuracy_summary")
        tf.summary.scalar("train_accuracy", self.tf_train)

        self.tf_valid = tf.placeholder(tf.float32, name="valid_accuracy_summary")
        tf.summary.scalar("valid_accuracy", self.tf_valid)

        # TODO: define more metrics you want to plot during training (e.g. training/validation accuracy)
             
        self.performance_summaries = tf.summary.merge_all()

    def write_episode_data(self, episode, eval_dict):
        
       # TODO: add more metrics to the summary 
       #with self.sess.graph.as_default():
       summary = self.sess.run(self.performance_summaries, feed_dict={self.tf_train: eval_dict["train"], self.tf_valid: eval_dict["valid"]})

       self.tf_writer.add_summary(summary, episode)
       self.tf_writer.flush()

    def close_session(self):
        self.tf_writer.close()
        self.sess.close()
