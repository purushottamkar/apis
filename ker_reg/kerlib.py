import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator
import time

def f(X,choice):
    global name
    if choice == 0:
        name='sin(x)'
        return np.sum(np.sin((np.pi/2)*X),axis=1)
    elif choice == 1:
        name='x*sin(3x)'
        return np.sum((X)*np.sin(3*X),axis=1)
    elif choice == 2:
        return np.sum(np.square(X/2)*np.sin(2*X),axis=1)
    elif choice == 3:
        #name='x(x-1)(x+1)/100'
        #return np.sum(X*(X-1)*(X+1)/50,axis=1)
        name='x'
        return np.sum(X,axis=1)
    elif choice == 4:
        name='(x-0.5)(x+0.5)(x-1)(x+1)'
        return np.sum((X-0.5)*(X+0.5)*(X-1)*(X+1),axis=1)
    elif choice == 5:
        name='x(x-2)(x+2)(x-4)(x+4)/50'
        return np.sum(X*(X-2)*(X+2)*(X-4)*(X+4)/50,axis=1)
    
def HT(t,frac):
    r=np.zeros_like(t)
    k=int(len(t)*frac)
    if k>0:
        ind=np.argpartition(abs(t),-k)[-k:]
        r[ind]=t[ind]
    return r

def getAllPairsDistances( A, B ):
    squaredNormsA = np.square( linalg.norm( A, axis = 1 ) )
    squaredNormsB = np.square( linalg.norm( B, axis = 1 ) )
    return squaredNormsA[:, np.newaxis] + squaredNormsB - 2 * A.dot( B.T )

# def gaussian_gram(X,h):
#     T=np.matmul(X,X.transpose())
#     G=(np.diag(T).reshape(-1,1)+np.diag(T))-2*T
#     return np.exp(-G/(2*h**2))

def get_response(G,w_star):
    n=G.shape[0]
    s=w_star.shape[0]
    _,V=linalg.eigh(G,eigvals=(n-s,n-1))
    V=np.fliplr(V)
    alpha_star=np.matmul(V,w_star)
    return np.matmul(G,alpha_star), alpha_star

def get_corruption(n,frac_k, sigma, corr_type='sym'):
    k=int(n*frac_k)
    b_star = np.zeros(n)
    ind=np.random.choice(n,k,replace=False)
    if corr_type=='sym':
        b_star[ind] = np.random.normal(loc=0, scale=sigma, size=(k,))
    else:
        b_star[ind] = abs(np.random.normal(loc=0, scale=sigma, size=(k,)))
    return b_star, ind


def get_adv_corr3(t, frac_k):
    n=t.shape[0]
    k=int(n*frac_k)
    sign=2*(t>0)-1
    corr=-sign*abs(t)
    dist= abs(t)/np.sum(abs(t))
    c_dist=[np.sum(dist[:i+1]) for i in range(n)]
    corr_ind=np.zeros(k, dtype=int)
    for i in range(k):
        unif=np.random.uniform()
        for j in range(n):
            if unif<c_dist[j]:
                corr_ind[i]=j
                break
    y=np.copy(t)
    y[corr_ind]=corr[corr_ind]
    return y, corr_ind


def NW(X_test,X,y,h):
    G=np.exp(-getAllPairsDistances(X_test,X)/(2*h**2))
    G=G/np.sum(G, axis=1).reshape(-1,1)
    return np.matmul(G,y)

def get_alpha(a,G,s):
    n=G.shape[0]
    if s>n:
        s=n
    e,V=linalg.eigh(G,eigvals=(n-s,n-1))
    e,V=np.flipud(e), np.fliplr(V)
    w_hat=np.matmul(V.transpose(),a)/e
    return w_hat, np.matmul(V,w_hat)

class APIS(BaseEstimator):
	# w_star is used only to plot convergence curves and not tune the method or decide stopping criterion in any way
	def __init__( self, frac_k = 0.0, frac_s = 1.0, h = 1.0 ):
		self.frac_k = frac_k
		self.frac_s = frac_s
		self.h = h

	def fit( self, X, y, a_star = None, max_iter = 20 ):
		self.X = X
		n = X.shape[0]
		start=time.time()
		G = np.exp( -getAllPairsDistances( X, X ) / (2 * self.h**2) )
		b = np.zeros(n)
		_, V = linalg.eigh( G, eigvals = ( int(n - self.frac_s * n), n - 1 ) )
		V = np.fliplr(V)
		
		if a_star is not None:
			loss = np.zeros( max_iter )
			clock = np.zeros( max_iter )
		
		for i in range( max_iter ):
			a = np.matmul( V, np.matmul( V.transpose(), y-b ) )
			b = HT( y-a, self.frac_k )
			if a_star is not None:
				loss[i] = np.linalg.norm( a - a_star , 2 )/np.sqrt(n)
				clock[i] = time.time()-start
			
		self.a = a
		self.alpha = np.matmul( np.linalg.pinv(G), a )
		if a_star is not None:
			self.loss = loss
			self.clock = clock            
		return self
	
	def predict( self, X ):
		G = np.exp( -getAllPairsDistances( X, self.X ) / (2 * self.h**2) )
		return np.matmul( G, self.alpha )
	
	# Return negative RMSE as score since sklearn assumes that score is something to be maximized
	def score( self, X, y ):
		n_test = X.shape[0]
		res = y - self.predict(X)
		res_corr = HT( res, self.frac_k )
		return -np.linalg.norm( res - res_corr, 2) / np.sqrt( int( n_test * ( 1 - self.frac_k ) ) )