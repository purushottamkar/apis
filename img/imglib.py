import numpy as np
import cv2


def noiseSP(img, frac_k, noise_type):
    shape=img.shape
    n=shape[0]*shape[1]
    k=int(n*frac_k)
    loc=np.unravel_index(np.random.choice(n,k, replace=False),shape)
    
    if noise_type ==0:
        img[loc]=0
    elif noise_type ==1:
        img[loc]=255
    elif noise_type ==2:
        img[loc]=255*np.random.choice(2,k, replace=True)
    elif noise_type ==3:        
        img[loc]=255*(img[loc]<127)
    
    return img

def rescale(img):
    img=img-np.min(img)
    mx=np.amax(img)
    mn=np.amin(img)
    
    if mx>mn:
        img=255*(img-mn)/(mx-mn)
    return img.astype(np.uint8)

def obfuscation(img,frac_k, loc_x, loc_y):
    shape=img.shape
    l=int(np.sqrt(shape[0]*shape[1]*frac_k))
    img[loc_x:loc_x+l, loc_y:loc_y+l]=img.min()
    return img

def HT(img,frac):
    ht_img=np.zeros_like(img)
    k=int(img.shape[0]*img.shape[1]*frac)
    if k>0:
        flat=abs(img).flatten(order='C')
        ind=np.argpartition(flat,-k)[-k:]
        ind=np.unravel_index(ind, img.shape, order='C')
        ht_img[ind]=img[ind]
    return ht_img


def flattenCoeffs(coeffs):
    flat=np.zeros(np.square(coeffs[-1][0].shape[0]*2))
    flat[0]=coeffs[0][0][0]
    l=len(coeffs)                    
    for i in range(1,l):
        size=np.power(2,2*(i-1))
        flat[size:2*size]=coeffs[i][0].flatten()
        flat[2*size:3*size]=coeffs[i][1].flatten()
        flat[3*size:4*size]=coeffs[i][2].flatten()
    return flat

def revertCoeffs(flat):
    l=int(np.log2(np.sqrt(len(flat))))
    r_coeffs=[]
    
    a=np.zeros((1,1))
    a[0][0]=flat[0]
    r_coeffs.append(a)
    
    for i in range(l):
        shape=(np.power(2,i),np.power(2,i))
        size=shape[0]*shape[1]
        a1=flat[size:2*size].reshape(shape)
        a2=flat[2*size:3*size].reshape(shape)
        a3=flat[3*size:4*size].reshape(shape)
        r_coeffs.append((a1,a2,a3))
    
    return r_coeffs

# HTW implements sparsity uniformly accross detailed coefficients
def HTW(coeffs,frac):
    flat=flattenCoeffs(coeffs)
    k=int(len(flat)*frac)
    ht_flat=np.zeros(len(flat))
    if k>0:
        ind=np.argpartition(abs(flat),-k)[-k:]
        ht_flat[ind]=flat[ind]
    return revertCoeffs(ht_flat)

# HTWS implements structural sparsity uniformly accross each frequency range
def HTWS(coeffs,frac):
    l=len(coeffs)
    for i in range(1,l):
        coeffs[i][0][:,:]=HT(coeffs[i][0],frac)
        coeffs[i][1][:,:]=HT(coeffs[i][1],frac)
        coeffs[i][2][:,:]=HT(coeffs[i][2],frac)
    return coeffs

# HTHWS implements structural sparsity keeping lower frequencies intact for wavelets
def HTHWS(coeffs,frac):
    coeffs[1][0][:,:]=HT(coeffs[1][0],frac)
    coeffs[1][1][:,:]=HT(coeffs[1][1],frac)
    coeffs[1][2][:,:]=HT(coeffs[1][2],frac)
    for i in range(2,len(coeffs)):
        coeffs[i][0][:,:]=np.zeros_like(coeffs[i][0])
        coeffs[i][1][:,:]=np.zeros_like(coeffs[i][1])
        coeffs[i][2][:,:]=np.zeros_like(coeffs[i][2])
    return coeffs

# HTSSP implements structural sparsity using power law in each frequency range of wavelets
def HTSSP(coeffs,alpha):
    l=len(coeffs)
    for i in range(1,l):
        frac=np.power(2,i*alpha)/np.power(2,i)
        coeffs[i][0][:,:]=HT(coeffs[i][0],frac)
        coeffs[i][1][:,:]=HT(coeffs[i][1],frac)
        coeffs[i][2][:,:]=HT(coeffs[i][2],frac)
    return coeffs