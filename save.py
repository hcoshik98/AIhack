import frame
import numpy as np
import encoding as en
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from tensorflow.keras.models import load_model
import pickle
from save_en import save

class CNN(object):

    def __init__(self, model_filepath):

        # The file path of model
        self.model_filepath = model_filepath
        # Initialize the model
        self.load_graph(model_filepath = self.model_filepath)


    def load_graph(self, model_filepath):
        '''
        Lode trained model.
        '''
        print('Loading model...')
        self.graph = tf.Graph()

        with tf.gfile.GFile(model_filepath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        for node in graph_def.node:
        	if node.op == 'RefSwitch':
        		node.op = 'Switch'
        		for index in range(len(node.input)):
        			if 'moving_' in node.input[index]:
        				node.input[index] = node.input[index] + '/read'
        	elif node.op == 'AssignSub':
        		node.op = 'Sub'
        		if 'use_locking' in node.attr: del node.attr['use_locking']
        	elif node.op == 'AssignAdd':
        		node.op='Add'
        		if 'use_locking' in node.attr: del node.attr['use_locking']

        print('Check out the input placeholders:')
        nodes = [n.name + ' => ' +  n.op for n in graph_def.node if n.op in ('Placeholder')]
        for node in nodes:
            print(node)

        with self.graph.as_default():
        	# Define input tensor
        	self.input = tf.placeholder(np.float32, shape = [None, 224, 224, 3], name='input')
        	self.dropout_rate = tf.constant(False, dtype=tf.bool, shape = [], name = 'phase_train')
        	self.dropout_rate = tf.constant(5, dtype=tf.int32, shape = [], name = 'batch_size')
        	tf.import_graph_def(graph_def, {'input': self.input,'phase_train':False, 'batch_size':5})
        self.graph.finalize()

        print('Model loading complete!')

        # Get layer names
        layers = [op.name for op in self.graph.get_operations()]
        for layer in layers:
            print(layer)
        
    
        self.sess = tf.Session(graph = self.graph)

    def test(self, data):

        # Know your output node name
        output_tensor = self.graph.get_tensor_by_name("import/embeddings:0")
        output = self.sess.run(output_tensor, feed_dict = {self.input: data, 'import/phase_train:0':0, 'import/batch_size:0':1})

        return output

C = CNN("./20190929-172255.pb")
save(C)

# [encode,name] = en.face_encode(C)
# frame.face_loc(C,encode,name)


	
	

	
