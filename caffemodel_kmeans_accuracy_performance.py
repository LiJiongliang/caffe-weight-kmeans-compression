import sys, subprocess
import numpy as np
sys.path.append("/home/jiongliang.li/caffe/python")
import caffe
import matplotlib.pyplot as plt

def main():

    net = caffe.Net("/home/jiongliang.li/caffe/models/bvlc_reference_caffenet/deploy.prototxt", caffe.TEST)
    kmeans_model = np.load("alexnet_caffmodel_weight.npz")
    print kmeans_model.files

    version, cbit, fbit = kmeans_model['compz_info']

    for item in net.params.items():
        name, layer = item
        newlabels = kmeans_model[name + '_weight_labels']
        codebook = kmeans_model[name + '_weight_codebook']
        newbias = kmeans_model[name + '_bias']
        if False:
            print newlabels.shape
            print codebook.shape
            print newbias.shape

        if "fc" in name:
            nbit = fbit
        elif "conv" in name:
            nbit = cbit

        #origin_size = net.params[name][0].data.flatten().size
        #weights_vec = np.empty(origin_size, dtype=np.float32)
        weights_tmp = []
        for i, val in enumerate(newlabels):
            weights_tmp.append(codebook[val])
        print len(weights_tmp)
        weights_vec = np.array(weights_tmp)

        newweights = weights_vec.reshape(net.params[name][0].data.shape)
        net.params[name][0].data[...] = newweights
        net.params[name][1].data[...] = newbias[...]
    net.save("alexnet_kmeans.caffemodel")
    if False:
        bin_command = "/home/jiongliang.li/caffe/build/tools/caffe.bin"
        solver = "--solver=/home/jiongliang.li/caffe/models/bvlc_reference_caffenet/solver.prototxt"
        model_caffe = "-weights "
        caffe_mode = "test"
        command = "%s %s %s" % (bin_command, caffe_mode, solver, model_caffe)
        p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        for line in p.stdout.readlines():
            print line.rstrip()
        retval = p.wait()

        if retval != 0:
            return

if __name__ == "__main__":
    main()