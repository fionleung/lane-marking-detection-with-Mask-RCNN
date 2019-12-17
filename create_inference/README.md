1. use [export_inference_graph.py](https://github.com/tensorflow/models/blob/master/research/object_detection/export_inference_graph.py.) to freeze the model

2. download [tf_text_graph_mask_rcnn.py](https://github.com/opencv/opencv/blob/master/samples/dnn/tf_text_graph_mask_rcnn.py)

    Run it to create a new .config for cv2.dnn

3. run mask_rcnn.py get  inference from image
run mask_rcnn_camera.py show viedo inference directly
run mask_runn_video.py  save inference as video


   [check reference here](https://www.pyimagesearch.com/2018/11/19/mask-r-cnn-with-opencv/)

    




