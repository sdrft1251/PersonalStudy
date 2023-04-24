import os
import tensorflow as tf



def transform(basedir, batch_size=256, input_shape=(1250, 1)):
    print("Change H5 models to TFLite models ---")
    q_class_model = tf.keras.models.load_model(os.path.join(basedir, "q_class.h5"))
    v_class_model = tf.keras.models.load_model(os.path.join(basedir, "v_class.h5"))
    s_class_model = tf.keras.models.load_model(os.path.join(basedir, "s_class.h5"))

    # Q class model save to pb with opt
    q_class_run_model = tf.function(lambda x: q_class_model(x))
    q_class_concrete_func = q_class_run_model.get_concrete_function(tf.TensorSpec([batch_size, input_shape[0], input_shape[1]], q_class_model.inputs[0].dtype))
    q_class_pb_path = os.path.join(basedir, "q_class_pb_tmp")
    q_class_model.save(q_class_pb_path, save_format="tf", signatures=q_class_concrete_func)
    # Converting Q class model to TFLite
    q_converter = tf.lite.TFLiteConverter.from_saved_model(q_class_pb_path)
    q_tflite_model = q_converter.convert()
    # Save
    q_tflite_model_path = os.path.join(basedir, "q_class.tflite")
    with open(q_tflite_model_path, 'wb') as f:
        f.write(q_tflite_model)
    
    # V class model save to pb with opt
    v_class_run_model = tf.function(lambda x: v_class_model(x))
    v_class_concrete_func = v_class_run_model.get_concrete_function(tf.TensorSpec([batch_size, input_shape[0], input_shape[1]], v_class_model.inputs[0].dtype))
    v_class_pb_path = os.path.join(basedir, "v_class_pb_tmp")
    v_class_model.save(v_class_pb_path, save_format="tf", signatures=v_class_concrete_func)
    # Converting V class model to TFLite
    v_converter = tf.lite.TFLiteConverter.from_saved_model(v_class_pb_path)
    v_tflite_model = v_converter.convert()
    # Save
    v_tflite_model_path = os.path.join(basedir, "v_class.tflite")
    with open(v_tflite_model_path, 'wb') as f:
        f.write(v_tflite_model)

    # S class model save to pb with opt
    s_class_run_model = tf.function(lambda x: s_class_model(x))
    s_class_concrete_func = s_class_run_model.get_concrete_function(tf.TensorSpec([batch_size, input_shape[0], input_shape[1]], s_class_model.inputs[0].dtype))
    s_class_pb_path = os.path.join(basedir, "s_class_pb_tmp")
    s_class_model.save(s_class_pb_path, save_format="tf", signatures=s_class_concrete_func)
    # Converting S class model to TFLite
    s_converter = tf.lite.TFLiteConverter.from_saved_model(s_class_pb_path)
    s_tflite_model = s_converter.convert()
    # Save
    s_tflite_model_path = os.path.join(basedir, "s_class.tflite")
    with open(s_tflite_model_path, 'wb') as f:
        f.write(s_tflite_model)

    return q_tflite_model_path, v_tflite_model_path, s_tflite_model_path

