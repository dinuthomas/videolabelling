## Steps to run the YouTube-8M feature extraction graph

1.  Checkout the repository and follow
    [the installation instructions](https://github.com/google/mediapipe/blob/master/docs/getting_started/install.md)
    to set up MediaPipe, use MediaPipe(v0.8.6) to avoid errors.

    ```bash
    git clone https://github.com/google/mediapipe.git
    cd mediapipe
    git checkout 374f5e2e7e818bde5289fb3cffa616705cec6f73
    ```

2.  Download the PCA and model data.

    ```bash
    mkdir /tmp/mediapipe
    cd /tmp/mediapipe
    curl -O http://data.yt8m.org/pca_matrix_data/inception3_mean_matrix_data.pb
    curl -O http://data.yt8m.org/pca_matrix_data/inception3_projection_matrix_data.pb
    curl -O http://data.yt8m.org/pca_matrix_data/vggish_mean_matrix_data.pb
    curl -O http://data.yt8m.org/pca_matrix_data/vggish_projection_matrix_data.pb
    curl -O http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    tar -xvf /tmp/mediapipe/inception-2015-12-05.tgz
    ```

3.  Get the VGGish frozen graph.

    Note: To run step 3, you must have Python 2.7 or 3.5+ installed
    with the TensorFlow 2.5.0 package installed.

    ```bash
    # cd to the root directory of the MediaPipe repo
    cd -

    pip3 install tf_slim
    python -m mediapipe.examples.desktop.youtube8m.generate_vggish_frozen_graph
    ```
