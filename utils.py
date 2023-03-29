import numpy as np
from tensorflow.keras.models import load_model
from huggingface_hub import from_pretrained_keras
from tensorflow import keras
from keras import layers
import tensorflow as tf
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import math
from layers import BilinearUpSampling2D
import os

def depth_model():
    custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}
    print('Loading model...')
    #model = from_pretrained_keras("keras-io/monocular-depth-estimation", custom_objects=custom_objects, compile=False)
    model = tf.keras.models.load_model('/home/juansoto/Trabajo_VO/SupervisedVO/model/kitti.h5', custom_objects=custom_objects, compile=False)
    model.trainable = False
    print('Successfully loaded model...')
    return model

def text_to_list(path_file):
    lines = open(path_file, "r")
    lista = lines.read()
    lista = lista.replace("[", "")
    lista = lista.replace("]", "")
    lista = lista.replace(" ", "")
    lista = lista.replace("'", "").split(',')
    return lista


def unpack_images(root_path, names_list):
    # names_list = text_to_list(os.path.join(root_path, names_list))
    # print(names_list)
    i = 0
    j = 0
    for names in names_list:
        #print(root_path, names)
        path_file = os.path.join(root_path, names)
        path_file = os.path.join(path_file)
        img_ex = Image.open(path_file)
        img_ex = tf.keras.preprocessing.image.img_to_array(img_ex)
        tgt_start_idx = int(img_ex.shape[1] // 2)
        tgt_image = tf.concat([img_ex[:, :tgt_start_idx, :], img_ex[:, tgt_start_idx:, :]], axis=2)
        if i > 0:
            img_stack = tf.concat([img_stack, tf.expand_dims(tgt_image, axis=0)], axis=0)

        else:
            tgt_image = tf.expand_dims(tgt_image, 0)
            img_stack = tgt_image
        i += 1
        # if i%batch == 0 or i==(len(names_list)-1):
        #     if j ==0:
        #         img_cont = img_stack
        #     else:
        #         img_cont = tf.concat([img_cont, img_stack], axis = 0)
        #         img_stack = None
        #         print(img_cont.shape)
        #         print("Concatenando")
        #     print(f"i: {i} ----- j: {j}")
        #     j += 1
    # print(f'Tamano de tensor: {img_stack.shape}')
    return img_stack



def loss_func(targets, predictions):
    reg_trans_loss = 1
    reg_rot_loss = 100
    rot_loss = tf.reduce_mean(abs(predictions[:3] - targets[:3])) * reg_rot_loss
    trans_loss = tf.reduce_mean(abs(predictions[3:] - targets[3:])) * reg_trans_loss
    total_loss = trans_loss + rot_loss

    return total_loss


def batch_generator(root_path, files, target_file, batch_size, type_calc=None):
    root_path = root_path.decode("utf-8")
    files = files.decode("utf-8")
    target_file = target_file.decode("utf-8")
    type_calc = type_calc.decode("utf-8")
    #print('################')
    #print(root_path, files, target_file, batch_size)
    #batch_size = 1 
    test = True
    files_list = text_to_list(os.path.join(root_path, files))
    len_file = len(files_list)
    target = np.load(os.path.join(root_path, target_file))
    num_batch = len_file // batch_size
    while test:
        if len_file == len(target):
            # print('Creating')
            for b in range(num_batch):
                # print(f'batch: {b}')
                if b < (num_batch - 1):
                    # print(b)
                    # print(b*batch_size,batch_size*(b+1))
                    files = files_list[b * batch_size:batch_size * (b + 1)]
                    stack = unpack_images(root_path, files)
                    targets = target[b * batch_size:batch_size * (b + 1)]
                else:
                    # print(b)
                    # print(b*batch_size,batch_size*(b))
                    files = files_list[b * batch_size:]
                    stack = unpack_images(root_path, files)
                    targets = target[b * batch_size:]
                    # test = False
                # print(len(files))

        else:
            print('Different length')
            break
        #stack = tf.squeeze(stack)
        
        if type_calc == 'tras':
            targets = targets[:, 3:6]
        if type_calc == 'rot':
            targets = targets[:, 0:3]
        
        # tensor_data = tf.data.Dataset.from_tensor_slices((stack, targets))
        yield (stack, targets)


def batch_generator_tf(root_path, files, target_file, type_calc=None):
    root_path = root_path.decode("utf-8")
    files = files.decode("utf-8")
    target_file = target_file.decode("utf-8")
    type_calc = type_calc.decode("utf-8")
    # print('################')
    # print(root_path, files, target_file, batch_size)
    # batch_size = 1
    test = True
    files_list = text_to_list(os.path.join(root_path, files))
    len_file = len(files_list)
    target = np.load(os.path.join(root_path, target_file))
    if len_file == len(target):
        print('Creating batches...')
        for file in files_list:
            # print(file)
            stack = unpack_images_tf(root_path, file)
            targets = target[files_list.index(file)]
            # print(f'Motion Vector: {targets.shape}')
            # print(targets)
            # print('###### Reshaping ######')
            targets = targets.reshape((1,6))
            # print(f'Motion Vector: {targets.shape}')
            # print(targets)


            if type_calc == 'tras':
                targets = targets[:, 3:6]
            if type_calc == 'rot':
                targets = targets[:, 0:3]
            
            
            yield (stack, targets)
    else:
        print('Different length')

def unpack_images_tf(root_path, filename):
    # names_list = text_to_list(os.path.join(root_path, names_list))
    # print(names_list)
    i = 0
    j = 0

    #print(root_path, names)
    path_file = os.path.join(root_path, filename)
    path_file = os.path.join(path_file)
    img_ex = Image.open(path_file)
    img_ex = tf.keras.preprocessing.image.img_to_array(img_ex)
    tgt_start_idx = int(img_ex.shape[1] // 2)
    tgt_image = tf.concat([img_ex[:, :tgt_start_idx, :], img_ex[:, tgt_start_idx:, :]], axis=2)
    # print(f'Stacked image: {tgt_image.shape}')
    return tgt_image


# batch_generator('./dest_folder','train_set_list.txt','poses_train.npy', batch_size = 1024)

def total_batches(root_path, file, batch_size):
    test = True
    files_list = text_to_list(os.path.join(root_path, file))
    len_file = len(files_list)
    num_batch = len_file // batch_size

    return num_batch


# print(total_batches('./dest_folder','train_set_list.txt', batch_size = 1024))

def plot_loss(model_history):
    history_dict = model_history.history
    loss_values = history_dict["loss"]
    val_loss_values = history_dict["val_loss"]
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, "bo", label="Training loss")
    plt.plot(epochs, val_loss_values, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def plot_dist_loss(history):
    epochs = history[0]
    train_loss = history[1]
    test_loss = history[2]

    plt.plot(epochs, loss_values, "bo", label="Training loss")
    plt.plot(epochs, val_loss_values, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def data_generator(root_path, files, target_file,strategy, batch_size = 8, type=None, op = 'train'):
    print('Generating Data')
    files_list = text_to_list(os.path.join(root_path, files))
    len_file = len(files_list)
    target = np.load(os.path.join(root_path, target_file))

    BUFFER_SIZE = int(len_file)

    BATCH_SIZE_PER_REPLICA = batch_size
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    EPOCHS = 10

    if len_file == len(target):
        files = files_list
        stack = unpack_images(root_path, files)

    else:
        print('Different length')

    if type == 'tras':
        target = target[:, 3:6]
    if type == 'rot':
        target = target[:, 0:3]
    if op == 'train':
        tensor_data = tf.data.Dataset.from_tensor_slices((stack, target)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)
    else:
        tensor_data = tf.data.Dataset.from_tensor_slices((stack, target)).batch(GLOBAL_BATCH_SIZE)

    tensor_data_p = strategy.experimental_distribute_dataset(tensor_data)

    return tensor_data_p

def plot_hist_list(e_his, t_loss, v_loss, model_dir):
    plt.scatter(e_his, v_loss, c='g', label='val_loss')
    plt.plot(e_his, t_loss, c='b', label='train_loss')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.title('train-val loss')
    plt.legend()
    plt.savefig(os.path.join(model_dir, 'training_his.png'))
    plt.show()
'''
t_loss = list(np.load(r'/home/juansoto/Trabajo_VO/Check_Res/train_loss.npy'))
v_loss = list(np.load(r'/home/juansoto/Trabajo_VO/Check_Res/val_loss.npy'))
e_his = [x for x in range(len(t_loss))]
model_dir = r'/home/juansoto/Trabajo_VO/Check_Res'
plot_hist_list(e_his, t_loss, v_loss, model_dir)
'''


def load_images_tensor(image_files):
    
    #image_files_d = tf.convert_to_tensor([image_files.read(i) for i in range(16)])
    image_files_d = image_files
    #print(f'STACK INPUT {image_files_d.get_shape()}')
    '''
    for id in range(image_files_d.get_shape().as_list()[0]):
        #print(id)
        #print('########')
        #print(tf.shape(img))
        #print(img.get_shape())
        file = tf.image.resize(image_files_d[id,:,:,:], (640, 480), method=tf.image.ResizeMethod.BICUBIC)
        file = file - tf.math.reduce_min(file)
        file = file / tf.math.reduce_max(file)
        #file = img
        #file = np.asarray(file, dtype="float32")
        #print(tf.math.reduce_min(file), tf.math.reduce_max(file))
        file = tf.expand_dims(file, axis = 0)
        if id == 0:
            final_stack = file
        else:
            final_stack = tf.concat([final_stack, file], axis = 0)
    '''
    file = tf.image.resize(image_files_d, (640, 480), method=tf.image.ResizeMethod.BICUBIC)
    file = file - tf.math.reduce_min(file)
    file = file / tf.math.reduce_max(file)
    final_stack = file
    #print(f'Finalshape: {final_stack.get_shape()}')
    
    return final_stack
    '''
    file = tf.image.resize(image_files_d, (640, 480), method=tf.image.ResizeMethod.BICUBIC)
    file = file - tf.math.reduce_min(file)
    file = file / tf.math.reduce_max(file)
    return file
    '''
def std_img(img):    
    img = img - tf.math.reduce_min(img)
    img = img / tf.math.reduce_max(img)
    
    return img

def depth_norm(x, maxDepth):
    return maxDepth / x


def predict(model, images, minDepth=10, maxDepth=1000, batch_size=2):
    # Support multiple RGBs, one RGB image, even grayscale
    if len(images.shape) < 3: images = tf.stack((images, images, images), axis=2)
    if len(images.shape) < 4: images = tf.reshape(images, (1, images.shape[0], images.shape[1], images.shape[2]))

    # Compute predictions
    #predictions = model.predict(images, batch_size=batch_size)
    predictions = model(images)    
    predictions = tf.expand_dims(predictions, axis = -1)
    # Put in expected range
    #return np.clip(depth_norm(predictions, maxDepth=maxDepth), minDepth, maxDepth) / maxDepth
    return tf.clip_by_value(depth_norm(predictions, maxDepth=maxDepth), minDepth, maxDepth) / maxDepth

def infer(model,image):
    #inputs = load_images_tensor(image)
    outputs = predict(model, image)
    #plasma = plt.get_cmap('plasma')
    rescaled = outputs[:,:, :, 0]
    rescaled = rescaled - tf.math.reduce_min(rescaled)
    rescaled = rescaled / tf.math.reduce_max(rescaled)
    #image_out = plasma(rescaled)[:, :, :3]
    #print(rescaled.get_shape(), tf.math.reduce_min(rescaled), tf.math.reduce_max(rescaled))
    return rescaled

'''
def infer_depth(model, input_batch):
    inputs = load_images_tensor(input_batch)
    depth_maps = []

    for id in range(inputs.get_shape().as_list()[0]):
        
        d_map = infer(model,inputs[id,:,:,:])
        
        d_map = tf.expand_dims(d_map, axis = -1)
        d_map = tf.image.resize(d_map, (204, 204), method=tf.image.ResizeMethod.BICUBIC)

        d_map = tf.expand_dims(d_map, axis =0)
        
        if id == 0:
            d_stack = d_map
        else:
            d_stack = tf.concat([d_stack, d_map], axis = 0)
                
    
    #print(f'DMAPS {d_stack.get_shape()}')

    return d_stack
''' 
def infer_depth(model, input_batch):
    inputs = load_images_tensor(input_batch)
    depth_maps = []

    d_map = infer(model,inputs)
    d_map = tf.image.resize(d_map, (204, 204), method=tf.image.ResizeMethod.BICUBIC)
                
    
    #print(f'DMAPS {d_map.get_shape()}')

    return d_map
    
    
def euler2mat(z, y, x):
  """Converts euler angles to rotation matrix
   TODO: remove the dimension for 'N' (deprecated for converting all source
         poses altogether)
   Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
  Args:
      z: rotation angle along z axis (in radians) -- size = [B, N]
      y: rotation angle along y axis (in radians) -- size = [B, N]
      x: rotation angle along x axis (in radians) -- size = [B, N]
  Returns:
      Rotation matrix corresponding to the euler angles -- size = [B, N, 3, 3]
  """
  B = z.shape[0]
  pi = tf.constant(math.pi)
  N = 1
  z = tf.clip_by_value(z, -pi, pi)
  y = tf.clip_by_value(y, -pi, pi)
  x = tf.clip_by_value(x, -pi, pi)

  # Expand to B x N x 1 x 1
  z = tf.expand_dims(tf.expand_dims(z, -1), -1)
  y = tf.expand_dims(tf.expand_dims(y, -1), -1)
  x = tf.expand_dims(tf.expand_dims(x, -1), -1)


  zeros = tf.zeros([B, N, 1, 1])
  ones  = tf.ones([B, N, 1, 1])

  cosz = tf.cos(z)
  sinz = tf.sin(z)
  rotz_1 = tf.concat([cosz, -sinz, zeros], axis=3)
  rotz_2 = tf.concat([sinz,  cosz, zeros], axis=3)
  rotz_3 = tf.concat([zeros, zeros, ones], axis=3)
  zmat = tf.concat([rotz_1, rotz_2, rotz_3], axis=2)

  cosy = tf.cos(y)
  siny = tf.sin(y)
  roty_1 = tf.concat([cosy, zeros, siny], axis=3)
  roty_2 = tf.concat([zeros, ones, zeros], axis=3)
  roty_3 = tf.concat([-siny,zeros, cosy], axis=3)
  ymat = tf.concat([roty_1, roty_2, roty_3], axis=2)

  cosx = tf.cos(x)
  sinx = tf.sin(x)
  rotx_1 = tf.concat([ones, zeros, zeros], axis=3)
  rotx_2 = tf.concat([zeros, cosx, -sinx], axis=3)
  rotx_3 = tf.concat([zeros, sinx, cosx], axis=3)
  xmat = tf.concat([rotx_1, rotx_2, rotx_3], axis=2)

  rotMat = tf.matmul(tf.matmul(xmat, ymat), zmat)
  return rotMat

def pose_vec2mat(vec):
  """Converts 6DoF parameters to transformation matrix
  Args:
      vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
  Returns:
      A transformation matrix -- [B, 4, 4]
  """

  batch_size, l = vec.get_shape().as_list()
  if l == 3:
    zeros = tf.zeros_like(vec, dtype=tf.float32)
    vec = tf.stack([vec, zeros], axis = 1)
    vec = tf.reshape(vec, (batch_size, 6))
  

  translation = tf.slice(vec, [0, 0], [-1, 3])
  translation = tf.expand_dims(translation, -1)
  rx = tf.slice(vec, [0, 3], [-1, 1])
  ry = tf.slice(vec, [0, 4], [-1, 1])
  rz = tf.slice(vec, [0, 5], [-1, 1])
  rot_mat = euler2mat(rz, ry, rx)
  rot_mat = tf.squeeze(rot_mat, axis=[1])
  filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
  filler = tf.tile(filler, [batch_size, 1, 1])
  transform_mat = tf.concat([rot_mat, translation], axis=2)
  transform_mat = tf.concat([transform_mat, filler], axis=1)

  return transform_mat






def is_int(array):
  b_var = True
  
  for e in array:
    b_var = e.is_integer()
    if b_var == False:
      return False
      break
  return True


def bilinear_f(grp, pnf, void, ref):
    
    i_p = grp[:,0,:]
    j_p = grp[:, 1, :]
    i_n = pnf[:,0,:]
    j_n = pnf[:,1,:]
    w = void.shape[2]
    h = void.shape[1]
    
        
    
    #i_n =  tf.case([(tf.less(i_n, 0), lambda: zero_p())], default=i_n)
    #j_n =  tf.case([(tf.less(j_n, 0), lambda: zero_p())], default=j_n)
    t_zero = tf.zeros_like(i_n, dtype=tf.float32)
    y_zero = tf.zeros_like(i_n, dtype=tf.float32)
    y_w = tf.ones_like(i_n, dtype=tf.float32)*(w-2)
    y_h = tf.ones_like(i_n, dtype=tf.float32)*(h-2)
    
    i_n =  tf.where(tf.less(i_n, y_zero), t_zero, i_n)
    j_n =  tf.where(tf.less(j_n, y_zero), t_zero, j_n)
    i_n =  tf.where(tf.math.greater(i_n, y_w), t_zero, i_n)
    j_n =  tf.where(tf.math.greater(j_n, y_h), t_zero, j_n)
    
    ir = tf.cast(i_p, dtype=tf.int32)
    jr = tf.cast(j_p, dtype=tf.int32)
    
    #print(tf.get_static_value(ir[0,:10], partial=False))
    #print(tf.get_static_value(jr[0,:10], partial=False))
    #print(tf.get_static_value(i_n[0,:10], partial=False))
    #print(tf.get_static_value(j_n[0,:10], partial=False))
    
    
    ref_c = tf.stack([jr,ir], axis= 1)
    #print(ref_c.shape)
    
    z = tf.constant([1,2,3], dtype=tf.int32)
    batch = void.shape[0]
    for b in range(batch):
        print(f'#### Batch {b} #####')
        p0 =ref[b,:,:,0]
        p1 = p0
        #p0 = tf.reshape(p0, (204*204,-1))
        print(p0.shape)
        
        #print(tf.transpose(ref_c[b]).shape)
        
        p0 = tf.gather_nd(p0, tf.transpose(ref_c[b]))
        #p0 = tf.reshape(p0, [204, 204])
        print(p0.shape)
        
        #p = ref[b,jr[b,:],ir[b,:],:]
    #_, ib = math.modf(i)
    #_, jb = math.modf(j)
        ib = tf.math.floor(i_n[b,:])
        jb = tf.math.floor(j_n[b,:])
        i1 = tf.cast(ib, dtype=tf.int32)
        j1 = tf.cast(jb, dtype=tf.int32)
        i2 = i1 + 1
        j2 = j1 + 1
    
        void0 = tf.scatter_nd(tf.transpose([j1,i1]), p0, [204,204])
        print(void[0].shape)
        #void[b, j1, i1,:].assign(p)
        #void[b, j2, i1,:].assign(p)
        #void[b, j1, i2,:].assign(p)
        #void[b, j2, i2,:].assign(p)
        void0 = tf.expand_dims(void0, axis = 0)
        
        if b == 0:
            '''
            print(p1[:5,5])
            print(p0[:10])
            print(ref_c[:3,:])
            print(j1[:10])
            print(i1[:10])
            print(void[:, :10, :10])
            '''
            void = void0
        else:
            void = tf.concat([void, void0], axis = 0)

        print(void.shape)
   
    return void

def multiply_K(cx, cy, batch):
    k = tf.convert_to_tensor(np.array([1, 0.000000e+00, cx, 0.000000e+00, 1,
    cy, 0.000000e+00, 0.000000e+00, 1.000000e+00]).reshape((3,3)))

    filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 4])
    k = tf.concat([k, tf.zeros([3, 1])], axis=1)
    k = tf.concat([k, filler], axis=0)
    k = tf.expand_dims(k, axis = 0)
    while k.shape[0] != batch:
        k = tf.concat([k,k], axis = 0)

    return k

def proj_custom_f(img, depth, M):
    print('PROJECT CUSTOM')
    
    
    batch = img.get_shape().as_list()[0]
    print(batch)
    Mr = M
    img_r = img
    #Mr = tf.convert_to_tensor([M.read(i) for i in range(batch)])
    #img_r = tf.convert_to_tensor([img.read(i) for i in range(batch)])
    h = 204
    w = 204
    
    if len(Mr.get_shape()) < 3:
        Mr = pose_vec2mat(Mr)
    print(tf.get_static_value(Mr))
   
    
    gr = gridf(depth, f_dims=True)
    

    k = multiply_K(int(w/2), int(h/2), batch)

    k_inv = tf.linalg.inv(k)

    d = tf.reshape(depth, [batch, 1, h,w])
    d = tf.math.abs(d - 1)

    void_img = tf.Variable(tf.zeros((batch, h, w, 3)))

    gr = gr.reshape(batch, 4, -1)

    s_p = tf.matmul(k_inv, gr)

    
    s_pn =  tf.multiply(s_p[:,:3,:],(1/tf.reshape(d, [batch, 1, -1])))
    s_pn = tf.concat([s_pn, tf.reshape(s_p[:,3,:],[batch, 1, -1])], axis = 1)
    p_n = tf.matmul(k, tf.matmul(Mr,s_p))

    p_n = tf.Variable(p_n)
    p_n = p_n/tf.reshape(p_n[:,2,:], [batch,1,-1])

    grf = gr[:,:2,:]
    p_nf = p_n[:,:2,:]
    
    print('To Bilinear')
    final_img = bilinear_f(grf, p_nf, void_img, img_r)
    
    
    return final_img

def pixel_error(img_r, img_ob, M, depth_model):
    depth = infer_depth(depth_model, img_r)
    projected_img = proj_custom_f(img_r, depth, M)

    error = tf.math.reduce_mean(img_ob - projected_img)

    return error


def projected_image(img_r,  M, depth_model):
    depth = infer_depth(depth_model, img_r)
    projected_img = proj_custom_f(img_r, depth, M)

    return projected_img
  
###############################################################
def gridf(depth, f_dims= False):
    #print('### GRID DEBUGGING ###')
    #print(depth.shape)
    batch = depth.shape[0]
    depth = depth[:,:,:,0]
    w = depth.shape[2]
    h = depth.shape[1]

    hg = tf.ones((1, h, w), dtype= tf.int32)

    # Creating  initial rows and columns
    a = tf.convert_to_tensor([x for x in range(0, w)], dtype=tf.int32)
    b = tf.convert_to_tensor([x for x in range(0, h)], dtype=tf.int32)
    # print(len(a), len(b))
    l = tf.stack([a, a])
    n = tf.stack([b, b])

    # print(l.shape, n.shape)

    for i in range(h - 2):
        l = tf.concat([l, tf.expand_dims(a, axis=0)], axis=0)


    for i in range(w - 2):
        n = tf.concat([n, tf.expand_dims(b, axis=0)], axis=0)

    n = tf.transpose(n)
    # print(l.shape, n.shape)


    I = tf.stack([l, n])
    I = tf.concat([I, hg], axis = 0)
    I = tf.concat([I, hg], axis=0)
    # print(I.shape, hg.shape)

    # print(I[0,:5, :5])
    # print(I[1, :5, :5])

    if f_dims:
        I = tf.expand_dims(I, axis=0)
        #I = tf.tile(I, [batch, 0])
        #print('Grid Test')
        I_n = tf.concat([I,I], axis = 0)
        while I_n.shape[0] != batch:
           I_n = tf.concat([I_n,I], axis = 0)
           #print(I_n.shape)

    return I_n


def get_pixel_value(img, x, y):
    """
    Utility function that composes a new image, with pixels taken
    from the coordinates given in x and y.
    The shapes of x and y have to match.
    The batch order is preserved.
    """

    # We assume that x and y have the same shape.
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    # Create a tensor that indexes into the same batch.
    # This is needed for gather_nd to work.
    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)
    return tf.gather_nd(img, indices)

def transport_pixel_value(img, x, y):
    """
    TODO
    """
    def batch_gen(b,h,w):
      r =tf.range(h*w*b, delta=1, dtype=None, name='range')
      r = tf.math.floordiv(r, (h*w))
      r = tf.reshape(r,(-1,1))
      return r

    img_f = tf.squeeze(tf.transpose(tf.reshape(img[:,:,:,0], (1,-1))))
    #print(f'$$$$$ IMAGE RESHAPED :{img_f.shape}')
    # We assume that x and y have the same shape.
    shape = tf.shape(x)
    batch = shape[0]
    h = shape[1]
    w = shape[2]
    b = batch_gen(batch, h, w)
    
    x_f = tf.reshape(x, (-1, 1))
    y_f = tf.reshape(y, (-1, 1))
    # Create a tensor that indexes into the same batch.
    # This is needed for gather_nd to work.
    indices = tf.transpose(tf.squeeze([b, x_f,y_f]))
    #print(f'INDICES SHAPE: {indices.shape}')
    t_img = tf.scatter_nd(indices, img_f, shape=(batch,h, w))

    return t_img

def dif_floor(tensor):
    x_floor_NOT_differentiable = tf.floor(tensor)
    x_floor_differentiable = tensor - tf.stop_gradient(tensor - x_floor_NOT_differentiable)
    return x_floor_differentiable

def bilinear_test(grp, pnf, img):
    #print('#### BILINEAR DEBUG ####')
    shape = tf.cast(tf.shape(img), dtype=tf.float32)
    batch = shape[0]
    h = shape[1]
    w = shape[2]

    i_p = grp[:,0,:]
    j_p = grp[:, 1, :]
    i_n = pnf[:,0,:]
    j_n = pnf[:,1,:]
   

    
    #i_n =  tf.case([(tf.less(i_n, 0), lambda: zero_p())], default=i_n)
    #j_n =  tf.case([(tf.less(j_n, 0), lambda: zero_p())], default=j_n)
    t_zero = tf.zeros_like(i_n, dtype=tf.float32)
    y_zero = tf.zeros_like(i_n, dtype=tf.float32)
    y_w = tf.ones_like(i_n, dtype=tf.float32)*(w-2)
    y_h = tf.ones_like(i_n, dtype=tf.float32)*(h-2)
    
    i_n =  tf.where(tf.less(i_n, y_zero), t_zero, i_n)
    j_n =  tf.where(tf.less(j_n, y_zero), t_zero, j_n)
    i_n =  tf.where(tf.math.greater(i_n, y_w), t_zero, i_n)
    j_n =  tf.where(tf.math.greater(j_n, y_h), t_zero, j_n)
    
    ir = tf.cast(i_p, dtype=tf.int32)
    jr = tf.cast(j_p, dtype=tf.int32)
    
    
    ref_c = tf.stack([jr,ir], axis= 1)

    #print(img.shape)
    #print(ir.shape, jr.shape)
    
    #z = tf.constant([1,2,3], dtype=tf.int32)
    
    ir1 = tf.reshape(ir,[batch, h,w])
    jr1 = tf.reshape(jr,[batch, h,w])
    

    p0 = get_pixel_value(img, ir1, jr1)
    #print(ir1.shape, jr1.shape)
    #print(f'P0 SHAPE: {p0.shape}')

    #ib = tf.math.floordiv(i_n, tf.constant(1, dtype= tf.float32))
    #jb = tf.math.floordiv(j_n, tf.constant(1, dtype= tf.float32))
    ib = dif_floor(i_n)
    jb = dif_floor(j_n)

    
    #i1 = tf.cast(ib, dtype=tf.int32)
    #j1 = tf.cast(jb, dtype=tf.int32)
    i1 = ib
    j1 = jb

    i2 = i1 + 1
    j2 = j1 + 1

    i1= tf.reshape(i1, [batch, h, w])
    j1= tf.reshape(j1, [batch, h, w])
    i2= tf.reshape(i2, [batch, h, w])
    j2= tf.reshape(j2, [batch, h, w])


    void0 = transport_pixel_value(p0, tf.cast(j1, dtype=tf.int32), tf.cast(i1, dtype=tf.int32))
    #void1 = transport_pixel_value(p0, j1, i2)
    #void2 = transport_pixel_value(p0, j2, i1)
    #void3 = transport_pixel_value(p0, j2, i2)
   
    #void = (void0 + void1 + void2 + void3)/4
    
    return void0
  
def multiply_K(cx, cy, batch):
    f = 100
    #k = tf.convert_to_tensor(np.array([f, 0.000000e+00, cx, 0.000000e+00, f,
    #cy, 0.000000e+00, 0.000000e+00, 1.000000e+00]).reshape((3,3)), dtype=tf.float32)
    
    k = tf.convert_to_tensor(np.array([9.799200e+02, 0.000000e+00, 6.900000e+02, 0.000000e+00, 9.741183e+02,
    2.486443e+02, 0.000000e+00, 0.000000e+00, 1.000000e+00]).reshape((3,3)), dtype=tf.float32)
 

    filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 4])
    k = tf.concat([k, tf.zeros([3, 1])], axis=1)
    k = tf.concat([k, filler], axis=0)
    k = tf.expand_dims(k, axis = 0)
    k_n = tf.concat([k,k], axis = 0)
    while k_n.shape[0] != batch:
        k_n = tf.concat([k_n,k], axis = 0)

    return k_n

def proj_test(img, depth, M):
    # print('### PROJECT_2 DEBUG START ###')
    depth = tf.image.resize(depth, [img.shape[1], img.shape[2]])
    batch = img.shape[0]
    #print(batch)
    if len(depth.shape) < 4:
      depth = tf.expand_dims(depth, axis = 0)
    gr = gridf(depth, f_dims=True)
    
    #shape = tf.shape(depth)
   
    batch = depth.shape[0]
    #print(batch)
    if len(M.shape) < 3:
        M = pose_vec2mat(M)

    

    h = depth.shape[1]
    w = depth.shape[2]
    # print(w,h)
    im1r = img
    # im1r = tf.image.resize(img, [h, w])

    k = multiply_K(int(w/2), int(h/2), batch)
    # if len(K.shape)<3:
    #     K = np.expand_dims(K, axis=0)
    # print(k)
    k_inv = tf.linalg.inv(k)
    
    ###########
    # d = depth[:,:,:,0].numpy().reshape((batch, 1, h,w))
    # d = np.abs(d - 1)
    # print(d.shape)
    d =tf.reshape(depth[:,:,:,0], (batch, 1, h,w))
    #d = 1-d
    #########

    void_img = tf.zeros((batch, h, w, 3))
    gr = tf.reshape(gr, (batch, 4, -1))
    # gr = gr.reshape(batch, 4, -1)
    gr = tf.cast(gr, dtype=tf.float32)

    s_p = tf.matmul(k_inv, gr)
    
    ####
    # s_p = s_p.numpy()
    # s_p[:,:3,:] =  tf.multiply(s_p[:,:3,:],(1/d.reshape(batch, 1, -1)))
    # s_p[:,:3,:] =  tf.multiply(s_p[:,:3,:],(1/tf.reshape(d, (batch, 1, -1))))

    #print(s_p[0,:,:5])
    epsilon = 1e-6
    div = 1/ (tf.reshape(d, (batch, 1, -1)) + epsilon)
    

    s1 = tf.multiply((tf.reshape(s_p[:,0,:], (batch, 1, -1))), div)
    s2 = tf.multiply((tf.reshape(s_p[:,1,:], (batch, 1, -1))), div)
    s3 = tf.multiply((tf.reshape(s_p[:,2,:], (batch, 1, -1))), div)
    s4 = tf.reshape(s_p[:,3,:], (batch, 1, -1))


    s_p1 = tf.concat([s1,s2], axis = 1)
    s_p1 = tf.concat([s_p1,s3], axis = 1)
    s_p1 = tf.concat([s_p1,s4], axis = 1)


    p_n = tf.matmul(k, tf.matmul(M,s_p1))

    # p_n = p_n.numpy()
    # p_n = p_n/p_n[:,2,:].reshape(batch,1,-1)
    z = 1/tf.reshape(p_n[:,2,:], (batch, 1, -1))
    p_n2 = tf.multiply(p_n, z)


    grp = gr[:,:2,:]
    pnf = p_n2[:,:2,:]
    pnf = tf.transpose(tf.reshape(pnf, (batch,2, h,w)), perm=[0,2,3,1])
    
    

    #rec = bilinear_test(grp, pnf, img)
    rec = bilinear_sampler(img, pnf)
    return rec

def bilinear_sampler(imgs, coords):
  """Construct a new image by bilinear sampling from the input image.
  Points falling outside the source image boundary have value 0.
  Args:
    imgs: source image to be sampled from [batch, height_s, width_s, channels]
    coords: coordinates of source pixels to sample from [batch, height_t,
      width_t, 2]. height_t/width_t correspond to the dimensions of the output
      image (don't need to be the same as height_s/width_s). The two channels
      correspond to x and y coordinates respectively.
  Returns:
    A new sampled image [batch, height_t, width_t, channels]
  """
  def _repeat(x, n_repeats):
    rep = tf.transpose(
        tf.expand_dims(tf.ones(shape=tf.stack([
            n_repeats,
        ])), 1), [1, 0])
    rep = tf.cast(rep, 'float32')
    x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
    return tf.reshape(x, [-1])

  with tf.name_scope('image_sampling'):
    coords_x, coords_y = tf.split(coords, [1, 1], axis=3)
    inp_size = imgs.get_shape()
    coord_size = coords.get_shape()
    out_size = coords.get_shape().as_list()
    out_size[3] = imgs.get_shape().as_list()[3]

    coords_x = tf.cast(coords_x, 'float32')
    coords_y = tf.cast(coords_y, 'float32')

    x0 = tf.floor(coords_x)
    x1 = x0 + 1
    y0 = tf.floor(coords_y)
    y1 = y0 + 1

    y_max = tf.cast(tf.shape(imgs)[1] - 1, 'float32')
    x_max = tf.cast(tf.shape(imgs)[2] - 1, 'float32')
    zero = tf.zeros([1], dtype='float32')

    x0_safe = tf.clip_by_value(x0, zero, x_max)
    y0_safe = tf.clip_by_value(y0, zero, y_max)
    x1_safe = tf.clip_by_value(x1, zero, x_max)
    y1_safe = tf.clip_by_value(y1, zero, y_max)

    ## bilinear interp weights, with points outside the grid having weight 0
    # wt_x0 = (x1 - coords_x) * tf.cast(tf.equal(x0, x0_safe), 'float32')
    # wt_x1 = (coords_x - x0) * tf.cast(tf.equal(x1, x1_safe), 'float32')
    # wt_y0 = (y1 - coords_y) * tf.cast(tf.equal(y0, y0_safe), 'float32')
    # wt_y1 = (coords_y - y0) * tf.cast(tf.equal(y1, y1_safe), 'float32')

    wt_x0 = x1_safe - coords_x
    wt_x1 = coords_x - x0_safe
    wt_y0 = y1_safe - coords_y
    wt_y1 = coords_y - y0_safe

    ## indices in the flat image to sample from
    dim2 = tf.cast(inp_size[2], 'float32')
    dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32')
    base = tf.reshape(
        _repeat(
            tf.cast(tf.range(coord_size[0]), 'float32') * dim1,
            coord_size[1] * coord_size[2]),
        [out_size[0], out_size[1], out_size[2], 1])

    base_y0 = base + y0_safe * dim2
    base_y1 = base + y1_safe * dim2
    idx00 = tf.reshape(x0_safe + base_y0, [-1])
    idx01 = x0_safe + base_y1
    idx10 = x1_safe + base_y0
    idx11 = x1_safe + base_y1

    ## sample from imgs
    imgs_flat = tf.reshape(imgs, tf.stack([-1, inp_size[3]]))
    imgs_flat = tf.cast(imgs_flat, 'float32')
    im00 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx00, 'int32')), out_size)
    im01 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx01, 'int32')), out_size)
    im10 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx10, 'int32')), out_size)
    im11 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx11, 'int32')), out_size)

    w00 = wt_x0 * wt_y0
    w01 = wt_x0 * wt_y1
    w10 = wt_x1 * wt_y0
    w11 = wt_x1 * wt_y1

    output = tf.add_n([
        w00 * im00, w01 * im01,
        w10 * im10, w11 * im11
    ])
    return output
