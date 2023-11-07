import numpy as np

class optimizer:
   """
   Class for fitting a Logistic Matrix Factorization model to a binary matrix of data
   using the ADAM optimizer
   """
   def __init__(self, Rij, D, c):
      """
      Inputs:
      Rij - binary data matrix. model bits for biases on the first axis of Rij only, so
            make sure the input is transposed accordingly
      D - embedding dimension
      c - factor by which positive observations are upweighted, for collaborative filtering
      """
      #input Data
      self.M = Rij.shape[0]; self.N = Rij.shape[1]; self.D = D
      self.Rij = Rij
      self.Wij = (c-1.0)*np.copy(Rij) + 1.0

      #initialize model variables
      self.Us = np.random.normal(size=(self.M,self.D))
      self.Vs = np.random.normal(size=(self.N,self.D))
      self.betas = np.random.normal(size=self.M)

      #ADAM parameters
      self.alpha = 0.001; self.t=0
      self.b1 = 0.9; self.b2=0.999; self.eps=1e-8
      #first and second gradient moment vectors
      self.Um = np.zeros((self.M, self.D)); self.Uv = np.zeros((self.M, self.D)) 
      self.Vm = np.zeros((self.N, self.D)); self.Vv = np.zeros((self.N, self.D))
      self.Bm = np.zeros(self.M); self.Bv = np.zeros(self.M) 

      #prior uncertainty parameters
      self.gamma_u = 0.5; self.gamma_v = 0.5;

      #Compute all model data and initialize loss
      self.prob_matrix()
      self.state = [self.loss()]

   def mask_test_data(self, test_idxs):
      """
      Only train on training data, set weights of test data to 0
      """
      self.Wij[test_idxs] = 0

   def prob_matrix(self):
      """
      Compute matrix of probabilities from logistic function of linear predictors
      """
      self.phi = self.Us@self.Vs.T + self.betas.reshape(-1,1)
      np.clip(self.phi, -100, 100, out=self.phi)
      self.pmat = 1.0/(1.0+np.exp(-1.0*self.phi))
        
   def likelihood(self):
      """
      Compute negative log likelihood of model
      """
      return np.sum(self.Wij*(np.log(1.0 + np.exp(self.phi)) - self.Rij*self.phi))
    
   def loss(self):
      """
      Compute negative log posterior
      """
      r_us = np.sqrt(np.sum(np.square(self.Us), axis=1)); r_vs = np.sqrt(np.sum(np.square(self.Vs), axis=1))
      U_prior = np.sum(self.gamma_u*np.square(r_us))
      V_prior = np.sum(self.gamma_v*np.square(r_vs))
      return self.likelihood() + U_prior + V_prior
    
   def u_gradient(self):
      """
      Loss gradient w.r.t. U vectors along M axis of Rij
      """
      self.ugrad = np.sum((self.Wij*(self.pmat - self.Rij)).reshape(self.M, self.N, 1)*self.Vs.reshape(1, self.N, self.D), axis=1) + 2.0*self.gamma_u*self.Us
        
   def v_gradient(self):
      """
      Loss gradient w.r.t. V vectors along N axis of Rij
      """
      self.vgrad = np.sum((self.Wij*(self.pmat - self.Rij)).reshape(self.M, self.N, 1)*self.Us.reshape(self.M, 1, self.D), axis=0) + 2.0*self.gamma_v*self.Vs
   
   def beta_gradient(self):
      """
      Loss gradient w.r.t. bias parameters along M axis of Rij
      """
      self.bgrad = np.sum(self.Wij*(self.pmat-self.Rij), axis=1) + self.betas

   def ADAM_step(self):
      """
      Single ADAM optimizer step. Compute time averaged first and second moments of
      gradient. Adaptively scale step size to account for average biases, then step down
      gradient.
      """
      self.t += 1
      #compute gradients
      self.prob_matrix()
      self.u_gradient()
      self.v_gradient()
      self.beta_gradient()
      #update moments
      self.Um = self.b1*self.Um + (1.0-self.b1)*self.ugrad
      self.Uv = self.b2*self.Uv + (1.0-self.b2)*np.square(self.ugrad)
      self.Vm = self.b1*self.Vm + (1.0-self.b1)*self.vgrad
      self.Vv = self.b2*self.Vv + (1.0-self.b2)*np.square(self.vgrad)
      self.Bm = self.b1*self.Bm + (1.0-self.b1)*self.bgrad
      self.Bv = self.b2*self.Bv + (1.0-self.b2)*np.square(self.bgrad)
      #compute stepsize and descend gradient
      at = self.alpha*np.sqrt(1-np.power(self.b2, self.t))/(1-np.power(self.b1, self.t))
      self.Us -= at*self.Um/(np.sqrt(self.Uv)+self.eps)
      self.Vs -= at*self.Vm/(np.sqrt(self.Vv)+self.eps)
      self.betas -= at*self.Bm/(np.sqrt(self.Bv)+self.eps)

   def train(self, Niter=15000):
      """
      Train model. Print progress update every Niter/20 iterations, and store loss every
      50 iterations.
      """
      dn = Niter/20
      print('Initial loss {}'.format(self.state[-1]))
      for n in np.arange(Niter):
         self.ADAM_step()
         if n%50==0:
            self.state.append(self.loss())
         if n%dn==0:
            print("iter: {}\t loss {}".format(n, self.state[-1]), flush=True)
                
   def test_prec_rec(self):
      """
      Compute precision and recall of test data on maximum likelihood model prediction.
      """
      self.prob_matrix()
      #get locations of test_data predictions and true labels
      test_ids = np.where(self.Wij == 0)
      test_labels = self.Rij[test_ids]
      pred_labels = (self.pmat > 0.5)[test_ids]
      #compute confusion values
      TP = np.sum(np.logical_and(pred_labels==1, test_labels==1))
      FP = np.sum(np.logical_and(pred_labels==1, test_labels==0))
      FN = np.sum(np.logical_and(pred_labels==0, test_labels==1))
      return(TP/(TP+FP), TP/(TP+FN))

   def test_AUC(self):
      """
      Compute area under PR curve on test data.
      """
      self.prob_matrix()
      #get locations of test_data predictions and true labels
      test_ids = np.where(self.Wij == 0)
      test_labels = self.Rij[test_ids]
      #number of true pos and negs
      P = np.sum(test_labels); N = test_labels.shape[0] - P
      TPR = []; FPR = []
      for p in np.arange(0,1.001, 0.001):
         #compute true and false positive rate at given prediction threshold
         pred = (self.pmat > p)[test_ids]
         TP = np.sum(np.logical_and(pred==1, test_labels==1))
         TN = np.sum(np.logical_and(pred==0, test_labels==0))
         TPR.append(TP/P); FPR.append(1.0 - TN/N)
      #compute integral of FPR, TPR curve
      TPR = np.asarray(TPR); FPR = np.asarray(FPR)
      srt_ids = np.argsort(FPR)
      FPR = FPR[srt_ids]; TPR = TPR[srt_ids]
      dFPR = np.diff(FPR); mTPR = np.asarray([0.5*(TPR[i]+TPR[i+1]) for i in np.arange(1000)])
      return np.sum(dFPR*mTPR)

   def gen_kfold_partition(self, k):
      """
      Generate k-fold partition of the data matrix for test train splits
      """
      #generate k random partitions of the number of indices
      Ntot=self.M*self.N
      binsize = int(Ntot/k)
      all_ids = np.arange(Ntot)
      part_ids = []
      for i in np.arange(k-1):
         samp = np.random.choice(all_ids, size=binsize, replace=False)
         part_ids.append(samp)
         all_ids = list(set(all_ids)-set(samp))
      part_ids.append(np.asarray(all_ids))
    
      #get array ids at all partitions
      all_locs = np.where(self.Rij > -1)
      parts = [(all_locs[0][part_ids[i]], all_locs[1][part_ids[i]]) for i in np.arange(k)]
    
      return parts
