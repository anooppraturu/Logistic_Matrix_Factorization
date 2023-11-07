import numpy as np
import LMF

M=200; N=250

print("Making synthetic data from generated probability matrix \n")
#generate synthetic data matrix from the generative model
generate_data = LMF.optimizer(np.zeros((M, N)), 2, 1.0)
#generate test data from both the maximum likelihood and random sample from generated probability matrix
example_Rmat_ML = (generate_data.pmat > 0.5).astype(int)
example_Rmat_sample = (generate_data.pmat > np.random.uniform(size=(M, N))).astype(int)

print("Training Model on maximum likelihood synthetic data \n")
ML_opt = LMF.optimizer(example_Rmat_ML, 2, 1.0)
parts = ML_opt.gen_kfold_partition(5)
ML_opt.mask_test_data(parts[0])
ML_opt.train()
print("ML test AUC: {}\n".format(ML_opt.test_AUC()))
bdiff = np.mean(np.square(ML_opt.betas-generate_data.betas))
print("mean square difference between model and fitted bias parameters:{}\n".format(np.mean(bdiff)))

print("Training Model on sampled synthetic data \n")
samp_opt = LMF.optimizer(example_Rmat_sample, 2, 1.0)
parts = ML_opt.gen_kfold_partition(5)
samp_opt.mask_test_data(parts[0])
samp_opt.train()
print("Sampled test AUC: {}\n".format(samp_opt.test_AUC()))
bdiff = np.mean(np.square(samp_opt.betas-generate_data.betas))
print("mean square difference between model and fitted bias parameters:{}\n".format(np.mean(bdiff)))
