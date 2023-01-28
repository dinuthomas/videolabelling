#!/bin/bash
source /opt/capstone/capstone/bin/activate
mkdir /tmp/mediapipe
cd /tmp/mediapipe
curl -O http://data.yt8m.org/pca_matrix_data/inception3_mean_matrix_data.pb
curl -O http://data.yt8m.org/pca_matrix_data/inception3_projection_matrix_data.pb
curl -O http://data.yt8m.org/pca_matrix_data/vggish_mean_matrix_data.pb
curl -O http://data.yt8m.org/pca_matrix_data/vggish_projection_matrix_data.pb
curl -O http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
tar -xvf /tmp/mediapipe/inception-2015-12-05.tgz

cd /opt/iproject/videolabelling/mediapipe
python -m mediapipe.examples.desktop.youtube8m.generate_vggish_frozen_graph

cd /opt/iproject/videolabelling/featuremap
sh start.sh ../mediapipe ../input_video ../inputfeats ../temp