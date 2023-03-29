import numpy as np
from tensorflow import keras
from keras import layers
import tensorflow as tf
import os
import sys
import modelos
import utils
import argparse
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Progbar
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, required=True, help="where the the data is  stored")
parser.add_argument("--model_dir", type=str, required=True, help="where the model is goint to be stored")
parser.add_argument("--model_name", type=str, required=True, help="model name")
parser.add_argument("--type_calc", type=str, required=True, help="tras or rot model")
args = parser.parse_args()


strategy = tf.distribute.MirroredStrategy(['/GPU:4', '/GPU:5', '/GPU:6', '/GPU:7'], cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

batch_size = 4

# train_gen = utils.data_generator(args.dataset_dir, 'train_set_list.txt', 'poses_train.npy', batch_size=batch_size,
#                                  type=args.type_calc, strategy=strategy)
# val_gen = utils.data_generator(args.dataset_dir, 'valid_set_list.txt', 'poses_test.npy', batch_size=batch_size,
#                                type=args.type_calc, strategy=strategy)

#train_gen = utils.batch_generator(args.dataset_dir, 'train_set_list.txt', 'poses_train.npy', batch_size=batch_size,
 #                                type=args.type_calc)

#val_gen = utils.batch_generator(args.dataset_dir, 'valid_set_list.txt', 'poses_test.npy', batch_size=batch_size,
 #                             type=args.type_calc)
train_calls = [args.dataset_dir, 'train_set_list.txt', 'poses_train.npy', args.type_calc]
val_calls = [args.dataset_dir, 'valid_set_list.txt', 'poses_test.npy', args.type_calc]

expected_shapes = (204, 204, 6),(1, 6)


train_gen = tf.data.Dataset.from_generator(utils.batch_generator_tf, args=train_calls, output_types= (tf.float32, tf.float32), output_shapes= (expected_shapes))
val_gen = tf.data.Dataset.from_generator(utils.batch_generator_tf, args=val_calls, output_types= (tf.float32, tf.float32), output_shapes= (expected_shapes))

train_gen = train_gen.batch(batch_size)
val_gen = val_gen.batch(batch_size)


### Definicion del modelo


fname = os.path.join(args.model_dir, "weights-{epoch:03d}-{val_loss:.4f}.h5")

callbacks_list = [
    keras.callbacks.ModelCheckpoint(filepath=fname, monitor="val_loss", save_best_only=True, save_freq="epoch",
                                    verbose=1),
    keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=10, verbose=0, mode="auto",
                                  baseline=None, restore_best_weights=False)
]

def  create_model():
    model = modelos.custom_Res((204, 204, 6), type_calc= args.type_calc)
    return model
    
tf.config.run_functions_eagerly(True)
with strategy.scope():
    loss_obj = tf.keras.losses.MeanAbsoluteError(reduction = tf.keras.losses.Reduction.NONE)
    #depth_model = utils.depth_model()
    

    def compute_loss(img_r,im_obj):
        #im_rec= utils.proj_custom_f(predictions[2], predictions[1], predictions[0])
        #im_rec = utils.projected_image(img_r,  M, depth_m)
        per_example_loss = loss_obj(img_r,im_obj)
        
       
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size= batch_size * strategy.num_replicas_in_sync)
        #return tf.reduce_sum(loss_obj(labels, predictions)) *(1. / (batch_size * strategy.num_replicas_in_sync))
        
    
    loss_map = keras.metrics.MeanAbsoluteError(name='train_loss')
    val_loss =  keras.metrics.MeanAbsoluteError(name='test_loss')
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    metrics = keras.metrics.MeanAbsoluteError()
    val_metrics = keras.metrics.MeanAbsoluteError()

    # model = modelos.custom_Res((204,204,6), type_calc = args.type_calc)
    # model.compile(optimizer = optimizer, loss = loss_map, metrics = metrics)
    model = modelos.get_model_2((204, 204, 6))


@tf.function
def distributed_train_step(dataset_inputs):
    per_replica_losses = strategy.run(train_step, args = (dataset_inputs,))

    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis = None)

def train_step(inputs):
    images, labels = inputs
    #labels = tf.squeeze(labels)
    im1 = utils.std_img(images[:,:,:,:3])
    im2 = utils.std_img(images[:,:,:,3:])
    # print(images.shape, labels.shape)
    print('###################')
    
    with tf.GradientTape() as tape:
        predictions = model(images, training = True)
        #print(predictions[0].shape, predictions[1].shape) 
        #print(tf.get_static_value(predictions[0]))
        '''
        print(predictions.shape) 
        print(tf.get_static_value(predictions[0]))

        M = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=True)
        im1t = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=True)
        for i in range(predictions.shape[0]):
            M = M.write(i, predictions[i,:])
            im1t = im1t.write(i, im1[i,:,:,:])
        '''
        loss = compute_loss(im2[:,:,:,0], predictions[1])
        print('#### LOSS ######')       
       
        print(tf.get_static_value(loss))
              
        

    gradients = tape.gradient(loss,  model.trainable_weights)
    print('gradients')
    optimizer.apply_gradients(zip(gradients,  model.trainable_weights))
    print('optimizer')
    loss_map.update_state(loss)

    return loss



@tf.function
def distributed_test_step(dataset_inputs):
  #per_replica_test_losses = strategy.run(test_step, args=(dataset_inputs,))
  #return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_test_losses, axis = None)
  return strategy.run(test_step, args=(dataset_inputs,))

def test_step(inputs):
  images, labels = inputs
  labels = tf.squeeze(labels)
  predictions = model(images, training=False)
  t_loss = compute_loss(predictions)

  val_metrics.update_state(t_loss)
  #val_metrics.update_state(predictions[2], predictions)


#print(model.summary())

EPOCHS = 100
t_loss =[]
v_loss =[]
e_his = []

total_batches = utils.total_batches(args.dataset_dir, 'train_set_list.txt', batch_size)
val_batches = utils.total_batches(args.dataset_dir, 'valid_set_list.txt', batch_size)
metric_name = 'Train loss'
for epoch in range(EPOCHS):
    print(f'<<<<<<<<<<< Iniciando Epoch {epoch} >>>>>>>>>>>')

    pb_i = Progbar(total_batches, stateful_metrics=metric_name)
    total_loss = 0.0
    num_batches = 0
    #train_gen = train_gen.take(total_batches)
    #print(total_batches)
    
    
    for batch in train_gen:
        batch_loss = distributed_train_step(batch)
        #print(batch_loss, batch[0].shape, tf.reduce_sum(batch[0])/1000000, batch[1][0,:])
        #pred = model(tf.expand_dims(batch[0][0,:,:,:], axis = 0), training = False)
        #print(batch[1][0,:], pred)
        total_loss += batch_loss
        num_batches += 1
        values = [('Train_loss', total_loss/(num_batches))]
        pb_i.update(num_batches, values=values)
        #print(num_batches)

    train_loss = total_loss / (num_batches)
    
    print(f'***** Validation epoch: {epoch} *****')
    val_batches = 0
    valid_loss = 0.0

    #val_gen = val_gen.take(val_batches)
    for batch in val_gen:
        #print(val_batches)
        distributed_test_step(batch)
        
    

    pb_i.update(num_batches, values=[('Train loss',train_loss),('Test loss', val_metrics.result())])
    template = ("Epoch {}, Loss: {}, train_MSE: {}, Test Loss: {}, Test MSE: {}")

    print(template.format(epoch, train_loss, loss_map.result(), val_metrics.result(),
                          val_metrics.result()))
    test_loss = val_metrics.result()
    t_loss.append(loss_map.result())
    v_loss.append(val_metrics.result())
    e_his.append(epoch)
    fname = os.path.join(args.model_dir,
                         'weights-{:02d}-{:.4f}'.format(epoch, test_loss))
    ## TODO Aplicar lr decay
  
    np.save(os.path.join(args.model_dir,'train_loss.npy'),t_loss)
    np.save(os.path.join(args.model_dir,'val_loss.npy'),v_loss)
    if len(v_loss) >= 2:
        if (min_v - v_loss[-1])/min_v >= 0.02:
            #model.save_weights(fname)
            model.save_weights(fname + ".h5")
            print(f'Saving model... Metric improved from {min_v:.4f} to {v_loss[-1]:.4f}')
            min_v = val_metrics.result()
        else:
            print('Model did not improve :(')
    else:
        #model.save_weights(fname)
        model.save_weights(fname + ".h5")
        print(f'Saving model... First update')
        min_v = val_metrics.result()

    #val_loss.reset_states()
    loss_map.reset_states()
    val_metrics.reset_states()

utils.plot_hist_list(e_his, t_loss, v_loss, model_dir=args.model_dir)
