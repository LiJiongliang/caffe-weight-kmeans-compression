import sys
import numpy as np
sys.path.append("/home/jiongliang.li/caffe/python")
sys.path.append("/home/jiongliang.li/deep_compression/KMeansRex/python")
import caffe
import KMeansRex
import matplotlib.pyplot as plt
import time

def main():

    net = caffe.Net("/home/jiongliang.li/caffe/models/bvlc_reference_caffenet/deploy.prototxt", caffe.TEST)
    net.copy_from("/home/jiongliang.li/caffe/_iter_200000.caffemodel")

    cbit = 8 # ref to songhan's paper conv quantization to 8bit
    fbit = 5 # ref to songhan's paper fc quantization to 5bit

    weights_dict = dict()
    weights_dict['compz_info'] = (1, int(cbit), int(fbit))

    for item in net.params.items():
        name, layer = item

        if "fc" in name:
            quantbit = int(fbit)
        elif "conv" in name:
            quantbit = int(cbit)

        # weights
        weights = net.params[name][0].data
        weights_vec = weights.flatten().astype(np.float64)
        weights_array = []
        for i, val in enumerate(weights_vec):
            weights_array.append([val])
        #print weights_array
        #dataset = np.reshape(weights_vec, -1)
        #weights_vec = np.ndarray(weights_array)
        print type(weights_array)
        #print weights_vec.shape

        # bias
        bias = net.params[name][1].data

        print "compressing layer", name, \
              "weight length" ,len(weights_vec), \
              "max of weights_vec", np.amax(weights_vec),\
              "min of weights_vec", np.amin(weights_vec)
        if False:
            plt.hist(weights_array, bins=1000)
            plt.xlabel(name)
            plt.ylabel('Probability')
            plt.show()

        #t_start = time.time()
        Mu, Z = KMeansRex.RunKMeans(weights_array, 2**quantbit)
        #t_stop = time.time()
        #kmeans_time = t_stop - t_start
        #print kmeans_time
        print Mu.shape
        print Z.shape
        #print codebook
        weights_dict[name + '_weight_labels'] = np.int8(Z)
        weights_dict[name + '_weight_codebook'] = np.float32(Mu)
        weights_dict[name + '_bias'] = np.float32(bias)
    np.savez("alexnet_caffmodel_weight.npz", **weights_dict)
    return

if __name__ == "__main__":
    main()


