caffe_root = 'home/socmgr/caffe/python/'

import sys
import json
sys.path.insert(0, caffe_root +'python')
import caffe

prototxt = '/home/socmgr/caffe/models/SqueezeNet/SqueezeNet_v1.1/deploy.prototxt'
caffemodel = '/home/socmgr/caffe/models/SqueezeNet/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel'

dict_params = {}
net = caffe.Net(prototxt,caffemodel, caffe.TEST)
for param in net.params:
	dict_params[param] = {'data': net.params[param][0].data.tolist(),
			      'shape': net.params[param][0].data.shape}

for param in net.params:
    for l in range(net.params[param][0].data.shape[0]):
         for k in range(net.params[param][0].data.shape[1]):
            for i in range(net.params[param][0].data.shape[2]):
                for j in range(net.params[param][0].data.shape[3]):
                    net.params[param][0].data[l][k][i][j]= ( float(int((net.params[param][0].data[l][k][i][j])*(2**15)))/(2**15) )
                    #print net.params[param][0].data[l][k][i][j]
net.save('f16.caffemodel')
