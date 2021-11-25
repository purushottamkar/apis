import numpy as np
import statsmodels.api as sm
from scipy.sparse.linalg import eigsh
import time
from sklearn.base import BaseEstimator


def HT(a,k):
    t=np.zeros(a.shape)
    if k==0:
        return t
    else:
        ind=np.argpartition(abs(a),-k, axis=None)[-k:]    
        t[ind,:]=a[ind,:]
    return t

class APIS(BaseEstimator):
	# w_star is used only to plot convergence curves and not tune the method or decide stopping criterion in any way
	def __init__( self, alpha = 0.0, w_init = None, w_star = None ):
		self.alpha = alpha
		self.w_init = w_init
		self.w_star = w_star

	def fit( self, X, y, max_iter = 50 ):
		start_time=time.time()
		self.clock = np.zeros( max_iter )
		n, d = X.shape
		k = int( n * self.alpha )
		a = np.matmul( X, self.w_init )
		self.l2=[]
		P = np.linalg.pinv(X)
		for i in range( max_iter ):
			b = HT( y - a, k )
			a = np.matmul( X, np.matmul( P, y - b ) )
			self.w = np.matmul( P, a )
			self.l2.append( np.linalg.norm( self.w - self.w_star ) )
			self.clock[i] = time.time() - start_time
			
		return self
	
	def predict( self, X ):
		return np.dot( X, self.w )
	
	# Return negative RMSE as score since sklearn assumes that score is something to be maximized
	def score( self, X, y ):
		n_test = X.shape[0]
		n_test_corr = int( self.alpha * n_test )
		res = y - self.predict(X)
		res_corr = HT( res, n_test_corr )
		return -np.linalg.norm( res - res_corr, 2) / np.sqrt( n_test - n_test_corr )

class STIR(BaseEstimator):
	# w_star is used only to plot convergence curves and not tune the method or decide stopping criterion in any way
	def __init__( self, eta = 1.01, alpha = 0.0, M_init = 10.0, w_init = None, w_star = None ):
		self.eta = eta
		self.alpha = alpha
		self.M_init = M_init
		self.w_init = w_init
		self.w_star = w_star
	
	def fit( self, X, y, max_iter = 40, max_iter_w = 1 ):
		start_time=time.time()
		n, d = X.shape
		M = self.M_init
		self.w = self.w_init
		
		self.l2=[]
		self.clock=[]
		itr=0
		
		while itr < max_iter:        
			iter_w = 0
			while iter_w < max_iter_w:
				s = abs( np.dot( X, self.w ) - y )
				np.clip( s, 1 / M, None, out = s )        
				s = 1/s
				
				mod_wls = sm.WLS( y, X, weights = s )
				res_wls = mod_wls.fit()
				self.w = res_wls.params.reshape( d, 1 )
				
				iter_w += 1     
				self.l2.append( np.linalg.norm( self.w - self.w_star ) )
				self.clock.append( time.time() - start_time )
							
				if iter_w >=max_iter_w:
					break
			itr += iter_w
			M *= self.eta
			
		return self
	
	def predict( self, X ):
		return np.dot( X, self.w )
	
	# Return negative RMSE as score since sklearn assumes that score is something to be maximized
	def score( self, X, y ):
		n_test = X.shape[0]
		n_test_corr = int( self.alpha * n_test )
		res = y - self.predict(X)
		res_corr = HT( res, n_test_corr )
		return -np.linalg.norm( res - res_corr, 2) / np.sqrt( n_test - n_test_corr )

class TORRENT(BaseEstimator):
	# w_star is used only to plot convergence curves and not tune the method or decide stopping criterion in any way
	def __init__( self, alpha = 0.0, w_init = None, w_star = None ):
		self.alpha = alpha
		self.w_init = w_init
		self.w_star = w_star

	
	def fit( self, X, y, max_iter = 6 ):
		start_time = time.time()
		n, d = X.shape    
		n_clean = int( ( 1 - self.alpha ) * n) # number of points we think are clean
		cleanIdx = np.arange(n)
		self.l2=[]
		self.clock=[]
		for t in range(max_iter):
			mod_ols = sm.OLS( y[cleanIdx], X[cleanIdx,:] )
			res_ols = mod_ols.fit()
			self.w = res_ols.params.reshape(d,1)
			
			res = abs( np.dot( X, self.w ) - y )
			cleanIdx = sorted( range( len( res ) ), key = lambda k: res[k] )[ 0: n_clean ]
			self.l2.append( np.linalg.norm( self.w - self.w_star ) )
			self.clock.append(time.time()-start_time)
			
		return self
	
	def predict( self, X ):
		return np.dot( X, self.w )
	
	# Return negative RMSE as score since sklearn assumes that score is something to be maximized
	def score( self, X, y ):
		n_test = X.shape[0]
		n_test_corr = int( self.alpha * n_test )
		res = y - self.predict(X)
		res_corr = HT( res, n_test_corr )
		return -np.linalg.norm( res - res_corr, 2) / np.sqrt( n_test - n_test_corr )