#!/bin/bash
mediapipedir=$1
src=$2
dst=$3
tempDir=$4

export PYTHONPATH=$mediapipedir:$PYTHONPATH

currrepodir=`pwd`

for file in ` ls $src `
do
	# change clip_end_time_sec to match the length of your video.
	# the output is generated is a mediasequence metadata from the input video at /tmp/mediapipe/metadata.pb
	echo "$src/$file"
	python3 -m mediapipe.examples.desktop.youtube8m.generate_input_sequence_example \
		  --path_to_input_video="$src/$file" \
		  --clip_end_time_sec=300
		  
	# moving the mediasequence metadata from generic /tmp location to given tempDir location
	mv /tmp/mediapipe/metadata.pb $tempDir/$file.metadata.pb
		  
	# changing to mediapipe workspace directory as bazel cannot be build 
	# The 'build' command is only supported from within a workspace (below a directory having a WORKSPACE file)
	cd $1
	echo `pwd`

	# # now the mediapipe binary is run to extract the features, creates a protobuf file $file.features.pb at given tempDir location
	bazel build -c opt --linkopt=-s \
		        --jobs=4 \
  			--define MEDIAPIPE_DISABLE_GPU=1 --define no_aws_support=true \
  			mediapipe/examples/desktop/youtube8m:extract_yt8m_features
	echo '*****************************#*'

       
        echo bazel-bin/mediapipe/examples/desktop/youtube8m/extract_yt8m_features \
                --calculator_graph_config_file=mediapipe/graphs/youtube8m/feature_extraction.pbtxt \
                --input_side_packets=input_sequence_example="$tempDir/$file.metadata.pb"  \
                --output_side_packets=output_sequence_example="$tempDir/$file.feature.pb"


	GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/autoflip/run_autoflip \
                --calculator_graph_config_file=mediapipe/graphs/youtube8m/feature_extraction.pbtxt \
                --input_side_packets=input_sequence_example="$tempDir/$file.metadata.pb"  \
                --output_side_packets=output_sequence_example="$tempDir/$file.feature.pb"

        # then featuremap.py is run to convert protobuf into json format
	python3 $currrepodir/featuremap.py "$tempDir/$file.feature.pb" $file $dst

	echo "Success!! - Feature Extraction done for :: $file"
done

echo "Done!"
