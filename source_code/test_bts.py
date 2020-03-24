# Library Imports
import os
import sys
import collections
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # noticed that all bts file had this

# Path Appending
ROOT_DIR = os.getcwd()
BTS_ROOT_DIR = os.path.join(ROOT_DIR, "bts")
BTS_MODEL_DIR = os.path.join(BTS_ROOT_DIR, "logs")
BTS_CUSTOM_LAYER_DIR = os.path.join(BTS_ROOT_DIR, 'tensorflow', 'custom_layer')

sys.path.append(BTS_ROOT_DIR)
sys.path.append(BTS_MODEL_DIR)
sys.path.append(BTS_CUSTOM_LAYER_DIR)

# Local Imports
from bts.tensorflow import bts_dataloader
from bts.tensorflow import bts

#-----------------------------------------------------------------------------------------------
# Main Code

# Creating the inputs for Btsdataloader and BtsModel

data_path = os.path.join(r"E:\MASTERS_STUFF\workspace\bts\dataset", "nyu_depth_v2", 'official_split', 'test')
filenames_file = r"E:\MASTERS_STUFF\workspace\filenames_file.txt"

checkpoints_path = r"E:\MASTERS_STUFF\workspace\bts\logs\bts_eigen_v2_pytorch_resnet50\model"

bts_parameters = collections.namedtuple('parameters', 
                                        'encoder, '
                                        'height, width, '
                                        'max_depth, '
                                        'batch_size, '
                                        'dataset, '
                                        'num_gpus, '
                                        'num_threads, '
                                        'num_epochs, ')

params = bts_parameters(encoder='resnet50_bts',
                        height=480,
                        width=640,
                        batch_size=None,
                        dataset="nyu",
                        max_depth=10,
                        num_gpus=1,
                        num_threads=None,
                        num_epochs=None)

# Creating BtsDataloader object
dataloader = bts_dataloader.BtsDataloader(data_path, None, filenames_file, params, 'test', do_kb_crop=False)

dataloader_iter = dataloader.loader.make_initializable_iterator()
iter_init_op = dataloader_iter.initializer
image, focal = dataloader_iter.get_next()

# Loading model
model = bts.BtsModel(params, 'test', image, None, focal=focal, bn_training=false)

# Creating session
config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=config)

# Initialize
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

# Saver ?
train_saver = tf.train.Saver()
train_saver.restore(sess, checkpoints_path)

# Running session
num_test_samples = 1
sess.run(iter_init_op)

# Model prediction
depth, pred_8x8, pred_4x4, pred_2x2 = sess.run([model.depth_est, model.lpg8x8, model.lpg4x4, model.lpg2x2])

depth = depth[0].squeeze()

# Formatting depth
depth = depth * 256.0 # Scaling depth
depth = depth.astype(np.uint16)

cv2.imshow("Output Depth", depth)
cv2.waitKey(0)
cv2.destroyAllWindows()
