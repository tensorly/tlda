import math
from scipy.stats import gamma

import tensorly as tl
from version0_20.cumulant_gradient import cumulant_gradient
import version0_20.tensor_lda_util as tl_util
from tensorly import check_random_state
import cupy as cp
import numpy as np

def loss_rec(factor, cumulant, theta):
    # cumulant = M3 - compute this
    rec = tl.cp_to_tensor((None, [factor]*3))
    rec_loss = -1* tl.tenalg.inner(rec, cumulant)
    # rec_loss = tl.norm(rec - cumulant, 2)**2
    ortho_loss = (1 + theta)/2*tl.norm(rec, 2)**2 # (1 + theta)/2
    return ortho_loss + rec_loss, ortho_loss, rec_loss/tl.norm(cumulant, 2)


class TLDA():
    def __init__(self, n_topic, alpha_0, n_iter_train, n_iter_test, batch_size, learning_rate, cumulant = None, gamma_shape = 1.0, smoothing = 1e-6,theta=1, seed=None): # we could try to find a more informative name for alpha_0
        # set all parameters here
        # r = check_random_state(1)
        cp.random.seed(seed)

        self.n_topic = n_topic
        self.alpha_0 = alpha_0
        self.n_iter_train = n_iter_train
        self.n_iter_test = n_iter_test
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma_shape = gamma_shape
        self.smoothing = smoothing
        self.theta     =  theta
        self.weights_ = tl.ones(self.n_topic)
        self.cumulant = cumulant
        
        # Initial values 
        log_norm_std = 1e-2
        log_norm_mean = alpha_0
        # order = 2 # always looking for the 3rd order moment
        # std_factors = (std/tl.sqrt(n_topic))**(1/order)
        # ensure initial values are on proper scale    
        # init_values = tl.abs(tl.tensor(cp.random.normal(0, std_factors, size=(n_topic, n_topic))))
        # tl.tensor(0.2*cp.ones((2, 2)))
        # mean = math.log((log_norm_mean**2)/math.sqrt(log_norm_mean**2 + log_norm_std))
        # std = math.log(1 + log_norm_std/(log_norm_mean**2))
        # deviation = tl.tensor(cp.random.lognormal(mean, std, size = (n_topic, n_topic)))
        # init_values = tl.abs(tl.eye(n_topic) - deviation)
        # init_values = deviation
        # init_values = tl.tensor(cp.random.uniform(-1, 1, size=(n_topic, n_topic)))
        # init_values, _ = tl.qr(init_values, mode='reduced')
        ortho_loss = 2
        i = 1
        while ortho_loss >= 1 and i <= 20:
            init_values = tl.tensor(cp.random.uniform(-1, 1, size=(n_topic, n_topic)))
            init_values, _ = tl.qr(init_values, mode='reduced')
            _, ortho_loss, _ = loss_rec(init_values, cumulant, theta)
            i += 1
        # init_values /= tl.norm(init_values, axis=0)
        # print(init_values)
        outFile = open("results/test_inits.txt", 'w')
        print(str(init_values), file=outFile)
        outFile.close()
        self.factors_ = init_values #/tl.norm(init_values, axis=0)

    def partial_fit(self, X_batch, learning_rate = None):
        '''Update the factors directly from the batch using stochastic gradient descent

        Parameters
        ----------
        X_batch : ndarray of shape (number_documents, num_topics) equal to the whitened
            word counts in each document in the documents used to update the factors

        verbose : bool, optional
            if True, print information about every 200th iteration
        '''
        # incremental version
        if learning_rate is None:
            learning_rate = self.learning_rate
        self.factors_ -= learning_rate*cumulant_gradient(self.factors_, X_batch, self.alpha_0,self.theta)

    def fit(self, X,verbose = True):
        '''Update the factors directly from X using stochastic gradient descent

        Parameters
        ----------
        X : ndarray of shape (number_documents, num_topics) equal to the whitened
            word counts in each document in the documents used to update the factors
        '''

        step_store = 1000
        trigger    = 0
        tol        = 1e-5 # self.learning_rate*1e-4
        i = 1
        train_iter = self.n_iter_train
        max_train_iter = 200 # 5000 # 50000
        max_diff_prev = tol+1
        max_diff = tol+1
        last_restart = 0
        max_diff_arr = []
        loss_arr = []
        ortho_loss_arr = []
        rec_loss_arr = []
        # next_iter = False
        
        # while i < train_iter: # and i < max_train_iter:
        #while (i - last_restart <= 2000 or max_diff >= tol) and i < max_train_iter:
        k = 0
        curr_max_step = tol + 1
        # curr_max_rec = None
        rec_loss = 1
        prev_rec_loss = 1
        prev_rec_greater = False
        converged = False
        last_init = 1
        # outFile = open("results/test_grads.txt", 'w')
        # while (i <= 100 or max_diff >= tol) and i < max_train_iter:
        # while (i <= 5000):
        # while (i <= 100 or max_diff <= max_diff_prev or max_diff >= tol) and i < max_train_iter:
        while (i <= 10 or max_diff >= tol) and i < max_train_iter:
        #while (i <= 10 or converged == False) and i < max_train_iter:
        #while (i <= 10 or curr_max_rec <= tol) and i < max_train_iter:
        # while (curr_max_step >= tol) and i < max_train_iter:
        #while (i <= 15 or rec_loss > prev_rec_loss or rec_loss < 0) and i < max_train_iter:
            #if i - last_restart > 2000:
            # if i == 2000:
            # prev_fac = tl.copy(self.factors_)
            prev_rec_loss = rec_loss
            if i == last_init+1:
                curr_ortho_loss = ortho_loss
            # max_diff_prev = max_diff
            #next_iter = False
            # outFile = open("results/test_grads.txt", 'w')
            # curr_max_step = 0
            # if curr_max_rec is not None:
            #     prev_max_rec = curr_max_rec
            # else:
            #     prev_max_rec = -1
            # curr_max_rec = -1
            for j in range(0, len(X), self.batch_size):
                prev_fac = tl.copy(self.factors_)
                y = X[j:j+self.batch_size]
                #lr = self.learning_rate*math.sqrt(10/(10+i))
                # if i <= 10:
                #     lr = self.learning_rate*1e-2
                # else:
                lr = self.learning_rate
                # if (tol< 1e-6*self.learning_rate and i >100) or trigger > 0:
                #     if trigger == 0:
                #         trigger = i
                #         print("trigger at iteration " + str(i))
                #     lr   = self.learning_rate*tl.sqrt(100/(100 + i - trigger))
                # step = lr*cumulant_gradient(self.factors_, y, self.alpha_0,self.theta)
                # curr_max = tl.max(abs(step))
                # if curr_max_step == None or curr_max >= curr_max_step:
                #     curr_max_step = curr_max

                # step /= tl.norm(step, axis=0)
                # gradient = cumulant_gradient(self.factors_, y, self.alpha_0, self.theta)
                # if curr_max_rec == -1 or max_rec > curr_max_rec:
                #     curr_max_rec = max_rec
                # self.factors_ -= lr*cumulant_gradient(self.factors_, y, self.alpha_0,self.theta)
                self.factors_ -= lr*cumulant_gradient(self.factors_, y, self.alpha_0, self.theta)
                self.factors_ /= tl.norm(self.factors_, axis=0)

                # if (tl.max(tl.abs(self.factors_ - prev_fac)) < tol):
                  #  converged = True
                  #  break
                # step =  lr*cumulant_gradient(self.factors_, y, self.alpha_0,self.theta)
                # if i <= 10:
                #     print(str(step), file=outFile)
                # self.factors_ -= step
                # if next_iter == False and tl.max(abs(step)) >= tol:
                #     next_iter = True
            #if i - last_restart > 2000:
            # if i == 2000:
            max_diff = tl.max(tl.abs(self.factors_ - prev_fac))
            max_diff_arr.append(float(max_diff))
            loss, ortho_loss, rec_loss = loss_rec(self.factors_, self.cumulant, self.theta)
            loss_arr.append(float(loss))
            ortho_loss_arr.append(float(ortho_loss))
            rec_loss_arr.append(float(rec_loss))
            rec_loss = -1*(float(rec_loss))

            # if i == last_init + 1 and ortho_loss > prev_ortho_loss:
            #     init_values = tl.tensor(cp.random.uniform(-1, 1, size=(self.n_topic, self.n_topic)))
            #     init_values, _ = tl.qr(init_values, mode='reduced')
            #     self.factors_ = init_values
            #     last_init = i

            #if i > 15 and rec_loss <= prev_rec_loss:
                #self.factors_ = tl.copy(prev_fac)
            #     outFile = open("results/final_diff.txt", 'w')
            #     print(max_diff, outFile)
            #     outFile.close()

            # if i - last_restart > 2000:
            #     self.factors_ = tl.tensor(cp.random.uniform(0, 1, size=(self.n_topic, self.n_topic)))
            #     last_restart = i
            # if max_diff < tol:
            #     if tl.max(tl.abs(tl.dot(self.factors_.T, self.factors_) - tl.eye(self.n_topic))) >= 0.7:
            #         self.factors_ = tl.tensor(cp.random.uniform(0, 1, size=(self.n_topic, self.n_topic)))
            #         max_diff = tol + 1
            #         last_restart = i
            # if max_diff < tol:
                
            # if (i % 10) == 0:
            #     # tol = abs(tl.norm(step)-step_store)/step_store
            #     # step_store = tl.norm(step) 
            #     if verbose == True:
            #         # print("Tolerance: " +str(tol))
            #         print("Epoch: " + str(i)+ " Gradient: " +str(step) )
            # if i == train_iter-1 and trigger == 0:
            #     train_iter += 1
            # elif i == train_iter-1 and (train_iter - trigger < 5000):
            #     train_iter += 5000 - (train_iter - trigger)
            i += 1
            # else:
            #     i += 1
        # outFile.close()
        outFile = open("results/max_diff_arr.txt", 'w')
        print(str(max_diff_arr), file=outFile)
        outFile.close()
        outFile = open("results/loss_arrs.txt", 'w')
        print(str(loss_arr), file=outFile)
        print(str(ortho_loss_arr), file=outFile)
        print(str(rec_loss_arr), file=outFile)
        outFile.close()
        print("total iterations: " + str(i))


    def _predict_topic(self, doc, adjusted_factor):
        '''Infer the document-topic distribution vector for a given document

        Parameters
        ----------
        doc : tensor of length vocab_size equal to the number of occurrences
                      of each word in the vocabulary in a document

        adjusted_factor : tensor of shape (number_topics, vocabulary_size) equal to the learned
               document-topic distribution

        Returns
        -------
        gammad : tensor of shape (1, n_cols) equal to the document/topic distribution
                 for the doc vector
        '''
        n_cols = len(self.weights_)

        gammad = tl.tensor(gamma.rvs(self.gamma_shape, scale= 1.0/self.gamma_shape, size = n_cols))
        exp_elogthetad = tl.tensor(np.exp(tl_util.dirichlet_expectation(gammad)))
        exp_elogbetad = tl.tensor(np.array(adjusted_factor))

        phinorm = (tl.dot(exp_elogbetad, exp_elogthetad) + 1e-100)
        mean_gamma_change = 1.0

        iter = 0
        while (mean_gamma_change > 1e-3 and iter < self.n_iter_test):
            lastgamma = tl.copy(gammad)
            gammad = ((exp_elogthetad * (tl.dot(exp_elogbetad.T, doc / phinorm))) + self.weights_)
            exp_elogthetad = tl.tensor(np.exp(tl_util.dirichlet_expectation(gammad)))
            phinorm = (tl.dot(exp_elogbetad, exp_elogthetad) + 1e-100)

            mean_gamma_change = tl.sum(tl.abs(gammad - lastgamma)) / n_cols
            all_gamma_change = gammad-lastgamma
            iter += 1

        return gammad

    def predict(self, X_test):
        '''Infer the document/topic distribution from the factors and weights and
        make the factor non-negative

        Parameters
        ----------
        X_test : ndarray of shape (number_documents, vocabulary_size) equal to the word
            counts in each test document

        Returns
        -------
        gammad_norm2 : tensor of shape (number_documents, number_topics) equal to
                       the normalized document/topic distribution for X_test

        factor : tensor of shape (vocabulary_size, number_topics) equal to the
                 adjusted factor
        '''

        adjusted_factor = tl.transpose(self.factors_)
        adjusted_factor = tl_util.non_negative_adjustment(adjusted_factor)
        adjusted_factor = tl_util.smooth_beta(adjusted_factor, smoothing=self.smoothing)

        gammad_l = (np.array([tl.to_numpy(self._predict_topic(doc, adjusted_factor)) for doc in X_test]))
        gammad_l = tl.tensor(np.nan_to_num(gammad_l))

        #normalize using exponential of dirichlet expectation
        gammad_norm = tl.tensor(np.exp(np.array([tl_util.dirichlet_expectation(g) for g in gammad_l])))
        gammad_norm2 = tl.tensor(np.array([row / np.sum(row) for row in gammad_norm]))

        return gammad_norm2, tl.transpose(self.factors_)
