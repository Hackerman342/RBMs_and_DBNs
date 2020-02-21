from util import *
from rbm import RestrictedBoltzmannMachine 
from dbn import DeepBeliefNet
# for removing files automatically
import os
import glob

if __name__ == "__main__":
    # Automatically remove files so network trains
    files_remove = True
    if files_remove:
        files = glob.glob('C:/Users/kwc57/Github_repos/RBMs_and_DBNs/trained_rbm/*')
        for f in files:
             os.remove(f)
        files = glob.glob('C:/Users/kwc57/Github_repos/RBMs_and_DBNs/trained_dbn/*')
        for f in files:
             os.remove(f)
         
    image_size = [28,28]
    train_imgs,train_lbls,test_imgs,test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)

    ''' restricted boltzmann machine '''
    '''
    print ("\nStarting a Restricted Boltzmann Machine..")

    rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
                                     #ndim_hidden=500,
                                     ndim_hidden=200,
                                     is_bottom=True,
                                     image_size=image_size,
                                     is_top=False,
                                     n_labels=10,
                                     batch_size=20
    )
    rbm.rf["period"] = 1
    err1 = rbm.cd1(visible_trainset=train_imgs, n_iterations=20)
    
    rbm2 = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
                                     ndim_hidden=300,
                                     is_bottom=True,
                                     image_size=image_size,
                                     is_top=False,
                                     n_labels=10,
                                     batch_size=20
    )
    
    err2 = rbm2.cd1(visible_trainset=train_imgs, n_iterations=20)
    rbm3 = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
                                     ndim_hidden=200,
                                     is_bottom=True,
                                     image_size=image_size,
                                     is_top=False,
                                     n_labels=10,
                                     batch_size=20
    )
    
    err3 = rbm3.cd1(visible_trainset=train_imgs, n_iterations=20)
    plt.figure()
    plt.plot(err1, label='500 nodes')
    plt.plot(err2, label='300 nodes')
    plt.plot(err3, label='200 nodes')
    plt.xlabel("Iterations")
    plt.ylabel("MSE")
    plt.title("Error over epochs")
    plt.legend()
    plt.show()
    '''
    #rbm.cd1(visible_trainset=train_imgs, n_iterations=15)
    #rbm.cd1(visible_trainset=train_imgs, n_iterations=20)
    
    ''' deep- belief net '''

    print ("\nStarting a Deep Belief Net..")
    
    #dbn = DeepBeliefNet(sizes={"vis":image_size[0]*image_size[1], "hid":500, "pen":500, "top":2000, "lbl":10},
    dbn = DeepBeliefNet(sizes={"vis":image_size[0]*image_size[1], "hid":200, "pen":200, "top":1000, "lbl":10},
                        image_size=image_size,
                        n_labels=10,
                        batch_size=10
    )
    
    ''' greedy layer-wise training '''

    #dbn.train_greedylayerwise(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=2000)
    dbn.train_greedylayerwise(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=2)
    '''
    # Plot MSE (like recon loss from first two DBN layers)
    plt.figure()
    plt.plot(dbn.MSE_v1, label='First RBM')
    plt.plot(dbn.MSE_v2, label='Second RBM')
    plt.xlabel("Epochs [All minibatches]")
    plt.ylabel("MSE")
    plt.title("Error over epochs | 784-500-500 Architecture")
    plt.legend()
    plt.show()
    '''
    dbn.recognize(train_imgs, train_lbls)
    
    dbn.recognize(test_imgs, test_lbls)

    for digit in range(10):
        digit_1hot = np.zeros(shape=(1,10))
        digit_1hot[0,digit] = 1
        dbn.generate(digit_1hot, name="rbms")

    ''' fine-tune wake-sleep training '''

    dbn.train_wakesleep_finetune(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=2000)

    dbn.recognize(train_imgs, train_lbls)
    
    dbn.recognize(test_imgs, test_lbls)
    
    for digit in range(10):
        digit_1hot = np.zeros(shape=(1,10))
        digit_1hot[0,digit] = 1
        dbn.generate(digit_1hot, name="dbn")
