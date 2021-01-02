import os
os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=cuda1,floatX=float32,optimizer=fast_compile'

from theano import function, config, shared, tensor
import numpy
import time
import matplotlib.pyplot as plt
from keras import optimizers
import argparse
import Models , LoadBatches
##################################################################################################################
# To check if GPU is used
vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], tensor.exp(x))
print(f.maker.fgraph.toposort())
t0 = time.time()
for i in range(iters):
    r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
if numpy.any([isinstance(x.op, tensor.Elemwise) and
              ('Gpu' not in type(x.op).__name__)
              for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')

####################################################################################################################




parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path",default="weights/ex1", type = str  )
parser.add_argument("--train_images",default="data/dataset1/images/images_prepped_train/", type = str  )
parser.add_argument("--train_annotations", default="data/dataset1/annotations/annotations_prepped_train/",type = str  )
parser.add_argument("--n_classes", default=12,type=int )
parser.add_argument("--input_height", type=int , default = 224  )
parser.add_argument("--input_width", type=int , default = 224 )

parser.add_argument('--validate',action='store_false')
parser.add_argument("--val_images", type = str , default = "data/dataset1/images/images_prepped_test/")
parser.add_argument("--val_annotations", type = str , default = "data/dataset1/annotations/annotations_prepped_test/")

parser.add_argument("--epochs", type = int, default = 50 )
parser.add_argument("--batch_size", type = int, default = 1 )
parser.add_argument("--val_batch_size", type = int, default = 1 )
parser.add_argument("--load_weights", type = str , default = "")

parser.add_argument("--model_name", type = str , default = "segnet")
parser.add_argument("--optimizer_name", type = str , default = "adadelta")
parser.add_argument("--learning_rate", type = float , default = 0.1)
parser.add_argument("--momentum", type = float , default = 0.9)

args = parser.parse_args()

train_images_path = args.train_images
train_segs_path = args.train_annotations
train_batch_size = args.batch_size
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
validate = args.validate
save_weights_path = args.save_weights_path
epochs = args.epochs
load_weights = args.load_weights
learning_rate=args.learning_rate
momentum=args.momentum

optimizer_name = args.optimizer_name
model_name = args.model_name

if validate:
	val_images_path = args.val_images
	val_segs_path = args.val_annotations
	val_batch_size = args.val_batch_size

model_select = { 'segnet':Models.Segnet.Segnet}
model_final = model_select[ model_name ]

m = model_final( n_classes , input_height=input_height, input_width=input_width)


#sgd = optimizers.SGD(lr=0.001 , momentum=0.9)
m.compile(loss='categorical_crossentropy',optimizer= 'adadelta' ,metrics=['accuracy'])


output_height = m.outputHeight
output_width = m.outputWidth



G  = LoadBatches.imageSegmentationGenerator( train_images_path , train_segs_path ,  train_batch_size,  n_classes , input_height , input_width , output_height , output_width   )



G2  = LoadBatches.imageSegmentationGenerator( val_images_path , val_segs_path ,  val_batch_size,  n_classes , input_height , input_width , output_height , output_width   )



if validate:
    print ("trainig")
    P=m.fit_generator(G,512,epochs=50)
    m.save_weights(save_weights_path+".h5")
    m.save(save_weights_path+".model"+".h5")  
else:
    print("validation")
    P=m.fit_generator(G2,512,epochs=50)
    m.save_weights(save_weights_path+".h5")
    m.save(save_weights_path+".model"+".h5")

logs = P.history

train_acc = logs['acc']
w1 = plt.figure(1)
line11, = plt.plot(train_acc, label='Training Accuracy', marker='o')
#line12, = plt.plot(train_loss, label='Testing Accuracy', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Epoch vs Accuracy')
plt.legend(loc=4)


train_loss = logs['loss']
w2 = plt.figure(2)
line21, = plt.plot(train_loss, label='Training Loss', marker='o')
#line22, = plt.plot(test_loss, label='Testing Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Epoch vs Loss')
plt.legend(loc=0)
plt.show()

