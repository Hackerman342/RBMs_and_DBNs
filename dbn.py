from util import *
from rbm import RestrictedBoltzmannMachine

class DeepBeliefNet():    

    ''' 
    For more details : Hinton, Osindero, Teh (2006). A fast learning algorithm for deep belief nets. https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    network          : [top] <---> [pen] ---> [hid] ---> [vis] 
                               `-> [lbl] 
    lbl : label
    top : top
    pen : penultimate
    hid : hidden
    vis : visible
    '''
    
    def __init__(self, sizes, image_size, n_labels, batch_size):

        """
        Args:
          sizes: Dictionary of layer names and dimensions
          image_size: Image dimension of data
          n_labels: Number of label categories
          batch_size: Size of mini-batch
        """

        self.rbm_stack = {
            
            'vis--hid' : RestrictedBoltzmannMachine(ndim_visible=sizes["vis"], ndim_hidden=sizes["hid"],
                                                    is_bottom=True, image_size=image_size, batch_size=batch_size),
            
            'hid--pen' : RestrictedBoltzmannMachine(ndim_visible=sizes["hid"], ndim_hidden=sizes["pen"], batch_size=batch_size),
            
            'pen+lbl--top' : RestrictedBoltzmannMachine(ndim_visible=sizes["pen"]+sizes["lbl"], ndim_hidden=sizes["top"],
                                                        is_top=True, n_labels=n_labels, batch_size=batch_size)
        }
        
        self.n_labels = n_labels
        
        self.sizes = sizes

        self.image_size = image_size

        self.batch_size = batch_size
        
        self.n_gibbs_recog = 15
        
        self.n_gibbs_gener = 200
        
        self.n_gibbs_wakesleep = 5

        self.print_period = 2000
        
        return


    def recognize(self,true_img,true_lbl):

        """Recognize/Classify the data into label categories and calculate the accuracy

        Args:
          true_imgs: visible data shaped (number of samples, size of visible layer)
          true_lbl: true labels shaped (number of samples, size of label layer). Used only for calculating accuracy, not driving the net
        """                
        
        lbl = np.ones(true_lbl.shape)/self.n_labels  # start the net by telling you know nothing about labels  
        #lbl = sample_categorical(lbl)
        
        # [TODO TASK 4.2] fix the image data in the visible layer and drive the network bottom to top. In the top RBM, run alternating Gibbs sampling \
        # and read out the labels (replace pass below and 'predicted_lbl' to your predicted labels).
        # NOTE : inferring entire train/test set may require too much compute memory (depends on your system). In that case, divide into mini-batches.
        # Get 1st hidden nodes from inputs (1st visible nodes)
        
        # Pass images through first layer 
        prob_h_1, h_1 = self.rbm_stack["vis--hid"].get_h_given_v_dir(true_img)
        # 2nd visible nodes are 1st hidden modes
        prob_v_2, v_2 = prob_h_1.copy(), h_1.copy()
        
        # Pass results from first layer through second layer
        prob_h_2, h_2 = self.rbm_stack["hid--pen"].get_h_given_v_dir(v_2)
        # 3rd visible nodes are 2nd hidden modes
        prob_v_3, v_3 = prob_h_2.copy(), h_2.copy() # To demonstrate what's occuring
        
        # Concatenate labels to end of penultimate layer
        v_3_all_nodes = np.concatenate((v_3, lbl), axis = 1)
        
        # Gibb's sample final RBM
        for i in range(self.n_gibbs_recog):
            _, h_3 = self.rbm_stack["pen+lbl--top"].get_h_given_v(v_3_all_nodes)
            _, v_3_all_nodes = self.rbm_stack["pen+lbl--top"].get_v_given_h(h_3)
                    
        # Predicted labels are last few columns of h_1 
        predicted_lbl = v_3_all_nodes[:, -self.n_labels:] # self.n_labels == true_lbl.shape[1]
            
        print ("accuracy = %.2f%%"%(100.*np.mean(np.argmax(predicted_lbl,axis=1)==np.argmax(true_lbl,axis=1))))
        
        return
    
    

    def generate(self,true_lbl,name):
        
        """Generate data from labels

        Args:
          true_lbl: true labels shaped (number of samples, size of label layer)
          name: string used for saving a video of generated visible activations
        """
        
        n_sample = true_lbl.shape[0]
        
        records = []        
        fig,ax = plt.subplots(1,1,figsize=(3,3))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.set_xticks([]); ax.set_yticks([])

        # [TODO TASK 4.2] fix the label in the label layer and run alternating Gibbs sampling in the top RBM. From the top RBM, drive the network \ 
        # top to the bottom visible layer (replace 'vis' from random to your generated visible layer).
        '''
        500‐unit layer can be initialized with either:
        1. a random sample (binomial distribution)
        2. a sample from biases
        3. a sample drawn from the distribution obtained by propagating a random image all the way form the input.
        '''
        # Using 1. a random sample (binomial distribution)
        #v_3_init = np.random.randint(2, size = (n_sample,self.sizes['pen']))
        # Using 3. a sample drawn from the distribution obtained by propagating
        # a random image all the way form the input.
        v_1_init = np.random.randint(2, size = (n_sample,self.sizes['vis']))
        _, v_2_init = self.rbm_stack["vis--hid"].get_h_given_v_dir(v_1_init)
        _, v_3_init = self.rbm_stack["hid--pen"].get_h_given_v_dir(v_2_init)
        
        # Concatenate binary samples to label
        v_3 = np.concatenate((v_3_init,true_lbl),axis = 1)
        
        for _ in range(self.n_gibbs_gener):
            # Ensure true label is fixed ("clamped") for every Gibb's sample
            v_3[:, -true_lbl.shape[1]:] = true_lbl
            # Get top hidden layer from true label and v_3
            prob_h_3, h_3 = self.rbm_stack["pen+lbl--top"].get_h_given_v(v_3)
            # Get v_3 back from h_3 (top hidden layer)
            prob_v_3, v_3 = self.rbm_stack["pen+lbl--top"].get_v_given_h(h_3)
            # Second hidden layer is top visible layer
            # Remove labels to go down to lower RBMs 
            h_2 = v_3[:, :-true_lbl.shape[1]]
            # Get v_2 from h_2
            prob_v_2, v_2 = self.rbm_stack["hid--pen"].get_v_given_h_dir(h_2)

            # First hidden layer is second visible layer
            h_1 = v_2
            # Get v_1 from h_1
            prob_v_1, v_1 = self.rbm_stack["vis--hid"].get_v_given_h_dir(h_1)

            # Give v_1 to vis for the animation
            vis = v_1
            
            records.append([ ax.imshow(vis.reshape(self.image_size), cmap="bwr", vmin=0, vmax=1, animated=True, interpolation=None) ] )
        #anim = stitch_video(fig,records).save("%s.generate%d.mp4"%(name,np.argmax(true_lbl)))            
            
        return

    def train_greedylayerwise(self, vis_trainset, lbl_trainset, n_iterations):

        """
        Greedy layer-wise training by stacking RBMs. This method first tries to load previous saved parameters of the entire RBM stack. 
        If not found, learns layer-by-layer (which needs to be completed) .
        Notice that once you stack more layers on top of a RBM, the weights are permanently untwined.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        try :

            self.loadfromfile_rbm(loc="trained_rbm",name="vis--hid")
            self.rbm_stack["vis--hid"].untwine_weights()            
            
            self.loadfromfile_rbm(loc="trained_rbm",name="hid--pen")
            self.rbm_stack["hid--pen"].untwine_weights()
            
            self.loadfromfile_rbm(loc="trained_rbm",name="pen+lbl--top")        

        except IOError :

            # [TODO TASK 4.2] use CD-1 to train all RBMs greedily
            
            
            ####################### First RBM #######################

            print ("training vis--hid")
             
            # CD-1 training for vis--hid 
            self.MSE_v1 = self.rbm_stack["vis--hid"].cd1(vis_trainset, n_iterations)
            
            # Save trained rbm and untwine weights for directed h + v         
            self.savetofile_rbm(loc="trained_rbm",name="vis--hid")
            self.rbm_stack["vis--hid"].untwine_weights()
            
            # Get 1st hidden nodes from inputs (1st visible nodes)
            prob_h_1, h_1 = self.rbm_stack["vis--hid"].get_h_given_v_dir(vis_trainset)
            
            ####################### Second RBM #######################

            # 2nd visible nodes are 1st hidden modes
            prob_v_2, v_2 = prob_h_1.copy(), h_1.copy()
            
            print ("training hid--pen")

            # CD-1 training for hid--pen 
            self.MSE_v2 = self.rbm_stack["hid--pen"].cd1(v_2, n_iterations)
                       
            # Save trained rbm and untwine weights for directed h + v         
            self.savetofile_rbm(loc="trained_rbm",name="hid--pen")            
            self.rbm_stack["hid--pen"].untwine_weights()
            
            # Get 2nd hidden nodes from 2nd visible nodes
            prob_h_2, h_2 = self.rbm_stack["hid--pen"].get_h_given_v_dir(v_2)
                       
            
            ####################### Third RBM #######################
            # 3rd visible nodes are 2nd hidden modes
            prob_v_3, v_3 = prob_h_2.copy(), h_2.copy() # To demonstrate what's occuring 
            
            print ("training pen+lbl--top")
            
            # Concatenate labels to end of penultimate layer
            v_3 = np.concatenate((v_3, lbl_trainset), axis = 1)
            
            # CD-1 training for pen+lbl--top 
            self.MSE_v3 = self.rbm_stack["pen+lbl--top"].cd1(v_3, n_iterations)
            
            # Save trained rbm | DO NOT untwine weights (leave h & v undirected)
            self.savetofile_rbm(loc="trained_rbm",name="pen+lbl--top")            

        return    



    def train_wakesleep_finetune(self, vis_trainset, lbl_trainset, n_iterations):

        """
        Wake-sleep method for learning all the parameters of network. 
        First tries to load previous saved parameters of the entire network.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """
        
        print ("\ntraining wake-sleep..")

        try :
            
            self.loadfromfile_dbn(loc="trained_dbn",name="vis--hid")
            self.loadfromfile_dbn(loc="trained_dbn",name="hid--pen")
            self.loadfromfile_rbm(loc="trained_dbn",name="pen+lbl--top")
            
        except IOError :            

            self.n_samples = vis_trainset.shape[0]

            for it in range(n_iterations):            
                                
                ####################### Wake-phase #######################
                # The unlabelled RBMs are already trained  
                
                # First RBN
                _, h_1 = self.rbm_stack["vis--hid"].get_h_given_v_dir(vis_trainset)
                _, v_1_pred = self.rbm_stack["vis--hid"].get_v_given_h_dir(h_1)
                
                self.rbm_stack["vis--hid"].update_generate_params(vis_trainset,h_1,v_1_pred)
                
                # Second RBN
                v_2 = h_1.copy()
                _, h_2 = self.rbm_stack["hid--pen"].get_h_given_v_dir(v_2)
                _, v_2_pred = self.rbm_stack["hid--pen"].get_v_given_h_dir(h_2)
                
                self.rbm_stack["hid--pen"].update_generate_params(v_2,h_2,v_2_pred)
               
                
                ####################### Undirected RBM training #######################
                v_3 = h_2.copy() # To demonstrate what's occuring
        
                v_3_all_nodes = np.concatenate((v_3, lbl_trainset), axis = 1)
                
                ''' # ONE OPTION
                self.rbm_stack["pen+lbl--top"].cd1(v_3_all_nodes, n_iterations=1)
            
                # Gibb's sample final RBM
                for _ in range(self.n_gibbs_wakesleep):  
                    _, h_3 = self.rbm_stack["pen+lbl--top"].get_h_given_v(v_3_all_nodes)
                    _, v_3_all_nodes = self.rbm_stack["pen+lbl--top"].get_v_given_h(h_3)
                    v_3_all_nodes[:, -self.n_labels:] = lbl_trainset  # We want to keep the labels clamped
                '''
                
                # SECOND OPTION
                v_3_init_gibbs = v_3_all_nodes.copy()
                _, h_3_init = self.rbm_stack["pen+lbl--top"].get_h_given_v(v_3_init_gibbs)
                
                h_3 = h_3_init.copy()
                
                for _ in range(self.n_gibbs_wakesleep):  
                    prob_v_3_all_nodes, v_3_all_nodes = self.rbm_stack["pen+lbl--top"].get_v_given_h(h_3)
                    v_3_all_nodes[:, -self.n_labels:] = lbl_trainset  # We want to keep the labels clamped
                    prob_h3, h_3 = self.rbm_stack["pen+lbl--top"].get_h_given_v(v_3_all_nodes)
                
                self.rbm_stack["pen+lbl--top"].update_params(v_3_init_gibbs, h_3_init, prob_v_3_all_nodes, prob_h3)   

                
                ####################### Sleep-phase #######################
                v_3_sleep = v_3_all_nodes[:-self.n_labels].copy()
                
                h_2_sleep = v_3_sleep.copy()
                _, v_2_sleep = self.rbm_stack["hid--pen"].get_v_given_h_dir(h_2_sleep)
                _, h_2_sleep_pred = self.rbm_stack["hid--pen"].get_h_given_v_dir(v_2_sleep)
                self.rbm_stack["hid--pen"].update_recognize_params(h_2_sleep, v_2_sleep, h_2_sleep_pred)
                
                
                h_1_sleep = v_2_sleep.copy()
                _, v_1_sleep = self.rbm_stack["vis--hid"].get_v_given_h_dir(h_1_sleep)
                _, h_1_sleep_pred = self.rbm_stack["vis--hid"].get_h_given_v_dir(v_1_sleep)
                self.rbm_stack["vis--hid"].update_recognize_params(h_1_sleep, v_1_sleep, h_1_sleep_pred)


                if it % self.print_period == 0 : print ("iteration=%7d"%it)
                        
            self.savetofile_dbn(loc="trained_dbn",name="vis--hid")
            self.savetofile_dbn(loc="trained_dbn",name="hid--pen")
            self.savetofile_rbm(loc="trained_dbn",name="pen+lbl--top")            

        return

    
    def loadfromfile_rbm(self,loc,name):
        
        self.rbm_stack[name].weight_vh = np.load("%s/rbm.%s.weight_vh.npy"%(loc,name))
        self.rbm_stack[name].bias_v    = np.load("%s/rbm.%s.bias_v.npy"%(loc,name))
        self.rbm_stack[name].bias_h    = np.load("%s/rbm.%s.bias_h.npy"%(loc,name))
        print ("loaded rbm[%s] from %s"%(name,loc))
        return
        
    def savetofile_rbm(self,loc,name):
        
        np.save("%s/rbm.%s.weight_vh"%(loc,name), self.rbm_stack[name].weight_vh)
        np.save("%s/rbm.%s.bias_v"%(loc,name),    self.rbm_stack[name].bias_v)
        np.save("%s/rbm.%s.bias_h"%(loc,name),    self.rbm_stack[name].bias_h)
        return
    
    def loadfromfile_dbn(self,loc,name):
        
        self.rbm_stack[name].weight_v_to_h = np.load("%s/dbn.%s.weight_v_to_h.npy"%(loc,name))
        self.rbm_stack[name].weight_h_to_v = np.load("%s/dbn.%s.weight_h_to_v.npy"%(loc,name))
        self.rbm_stack[name].bias_v        = np.load("%s/dbn.%s.bias_v.npy"%(loc,name))
        self.rbm_stack[name].bias_h        = np.load("%s/dbn.%s.bias_h.npy"%(loc,name))
        print ("loaded rbm[%s] from %s"%(name,loc))
        return
        
    def savetofile_dbn(self,loc,name):
        
        np.save("%s/dbn.%s.weight_v_to_h"%(loc,name), self.rbm_stack[name].weight_v_to_h)
        np.save("%s/dbn.%s.weight_h_to_v"%(loc,name), self.rbm_stack[name].weight_h_to_v)
        np.save("%s/dbn.%s.bias_v"%(loc,name),        self.rbm_stack[name].bias_v)
        np.save("%s/dbn.%s.bias_h"%(loc,name),        self.rbm_stack[name].bias_h)
        return
    
