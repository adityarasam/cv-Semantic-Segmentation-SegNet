import os
os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=cuda0,floatX=float32,optimizer=fast_compile'

from theano import function, config, shared, tensor
import numpy
import time
from keras import optimizers

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



import argparse
import Models , LoadBatches
from keras.models import load_model
import glob
import cv2
import numpy as np
import random

parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path",default="weights/ex1", type = str  )
parser.add_argument("--epoch_number", type = int, default = 0 )
parser.add_argument("--test_images", type = str , default="data/dataset1/images/images_prepped_test/")
parser.add_argument("--test_annotations", type = str , default="data/dataset1/images/annotations_prepped_test/")
parser.add_argument("--output_path", type = str , default = "output/")
parser.add_argument("--input_height", type=int , default = 224  )
parser.add_argument("--input_width", type=int , default = 224 )
parser.add_argument("--model_name", type = str , default = "segnet")
parser.add_argument("--n_classes", type=int, default=12 )

args = parser.parse_args()

n_classes = args.n_classes
model_name = args.model_name
images_path = args.test_images
input_width =  args.input_width
input_height = args.input_height
epoch_number = args.epoch_number
save_weights_path = args.save_weights_path

test_images_path = args.test_images
test_segs_path = args.test_annotations

modelFns = { 'segnet':Models.Segnet.Segnet}
modelFN = modelFns[ model_name ]

m = modelFN( n_classes , input_height=input_height, input_width=input_width   )
m.load_weights( args.save_weights_path+".model"+".h5")

#sgd = optimizers.SGD(lr=0.001 , momentum=0.9)
m.compile(loss='categorical_crossentropy',
      optimizer= 'adadelta',
      metrics=['accuracy'])



output_height = m.outputHeight
output_width = m.outputWidth

images = glob.glob( images_path + "*.jpg"  ) + glob.glob( images_path + "*.png"  ) +  glob.glob( images_path + "*.jpeg"  )
images.sort()

colors = [  ( random.randint(0,255),random.randint(0,255),random.randint(0,255)   ) for _ in range(n_classes)  ]

Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road_marking = [255,69,0]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

label_colours = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])





for imgName in images:
	outName = imgName.replace( images_path ,  args.output_path )
  
	X = LoadBatches.getImageArr(imgName , args.input_width  , args.input_height  )
	pr = m.predict( np.array([X]) )[0]
	pr = pr.reshape(( output_height ,  output_width , n_classes ) ).argmax( axis=2 )
	seg_img = np.zeros( ( output_height , output_width , 3  ) )
	for c in range(n_classes):
		seg_img[:,:,0] += ( (pr[:,: ] == c )*( label_colours[c][0] )).astype('uint8')
		seg_img[:,:,1] += ((pr[:,: ] == c )*( label_colours[c][1] )).astype('uint8')
		seg_img[:,:,2] += ((pr[:,: ] == c )*( label_colours[c][2] )).astype('uint8')
	seg_img = cv2.resize(seg_img  , (input_width , input_height ))
	cv2.imwrite(  outName , seg_img )
print("hello")