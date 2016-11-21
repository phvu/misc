from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import glob
import numpy as np
import pandas as pd
import tensorflow as tf


class NodeLookup(object):
    """Converts integer node ID's to human readable labels."""

    def __init__(self, label_file):
        self.node_lookup = self.load(label_file)

    def load(self, label_file):
        if not tf.gfile.Exists(label_file):
            tf.logging.fatal('File does not exist %s', label_file)
        with open(label_file, 'r') as f:
            labels = [s.strip() for s in f.readlines()]
        return dict(enumerate(labels))

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return node_id
        return self.node_lookup[node_id]


class ImageClassifier(object):

    def __init__(self, model_file, label_file, num_top_predictions=5, layer_name='final_result:0'):
        self.model_file = model_file
        self.num_top_predictions = num_top_predictions
        self.layer_name = layer_name
        self.node_lookup = NodeLookup(label_file)
        self.create_graph()

    def create_graph(self):
        """Creates a graph from saved GraphDef file and returns a saver."""
        # Creates graph from saved graph_def.pb.
        with tf.gfile.FastGFile(self.model_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

    def inference(self, images):

        with tf.Session() as sess:
            softmax_tensor = sess.graph.get_tensor_by_name(self.layer_name)

            for image in images:

                if not tf.gfile.Exists(image):
                    tf.logging.warn('File does not exist %s', image)
                    continue
                print(image)
                image_data = tf.gfile.FastGFile(image, 'rb').read()

                predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
                predictions = np.squeeze(predictions)

                top_k = predictions.argsort()[-self.num_top_predictions:][::-1]
                results = {}
                for node_id in top_k:
                    human_string = self.node_lookup.id_to_string(node_id)
                    score = float(predictions[node_id])
                    results[human_string] = score
                yield image, results


def main():
    model_file = sys.argv[1]
    label_file = sys.argv[2]
    image_dir = sys.argv[3]
    result_file = sys.argv[4]
    classifier = ImageClassifier(model_file=model_file,
                                 label_file=label_file,
                                 num_top_predictions=8,
                                 layer_name='final_result:0')
    results = []
    classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    for idx, (image, result) in enumerate(classifier.inference(sorted(glob.glob(os.path.join(image_dir, "*"))))):
        entry = {'image': os.path.split(image)[1]}
        entry.update((c, result[c.lower()]) for c in classes)
        results.append(entry)

        if idx % 50 == 0:
            pd.DataFrame(columns=['image'] + classes, data=results).to_csv(result_file, index=False)
    pd.DataFrame(columns=['image'] + classes, data=results).to_csv(result_file, index=False)


if __name__ == '__main__':
    main()
