import time

from sklearn import datasets
import numpy as np

diabetes = datasets.load_diabetes()
X=diabetes.data
y=diabetes.target.reshape(-1,1)

def soft_threshold(X, rho):
    '''
    soft threshold function
    '''
    zero = np.matrix(np.zeros(X.shape))
    return np.multiply(np.sign(X), np.maximum(np.abs(X)-rho, zero))

def objective_function(X, beta, y, lamda):
    return 1/2*np.linalg.norm(X.dot(beta)-y)**2+lamda*np.linalg.norm(beta,1)

def proximal(X,beta,y,lamda,iter_num):
    loss=[]
    c=np.linalg.norm(X)**2
    t=time.time()
    for j in range(iter_num):
        temp_beta=beta-lamda*np.matmul(X.T,(X.dot(beta)-y))
        beta_new=soft_threshold(temp_beta/c,lamda/c)
        loss_temp=objective_function(X,beta_new,y,lamda)
        loss.append(loss_temp)
        beta=beta_new
    t-=time.time()
    return loss

def acc_proximal1(X,beta,y,lamda,iter_num):
    loss = []
    c = np.linalg.norm(X) ** 2
    t = time.time()
    for j in range(1,iter_num+1):
        if j > 1:
            v = beta + (j - 2) / (j + 1) * (beta - beta_old)
            temp_beta = v - lamda * np.matmul(X.T, (X.dot(v) - y))
            beta_new = soft_threshold(temp_beta / c, lamda / c)
            loss_temp = objective_function(X, beta_new, y, lamda)
            loss.append(loss_temp)
            beta, beta_old = beta_new, beta
        else:
            temp_beta = beta - lamda * np.matmul(X.T, (X.dot(beta) - y))
            beta = soft_threshold(temp_beta / c, lamda / c)
            loss_temp = objective_function(X, beta, y, lamda)
            loss.append(loss_temp)
            beta_old = beta
    t -= time.time()
    return loss

def acc_proximal2(X,beta,y,lamda,iter_num):
    loss = []
    c = np.linalg.norm(X) ** 2
    t = time.time()
    k = 1
    v = beta
    for j in range(1,iter_num+1):
        beta_old = beta
        temp_beta = v - lamda * np.matmul(X.T, (X.dot(v) - y))
        beta = soft_threshold(temp_beta / c, lamda / c)
        k0=k
        k = (1 + np.sqrt(1 + 4*(k**2)))/2
        v = beta + ((k0 - 1)/k)*(beta - beta_old)
        loss_temp = objective_function(X, beta, y, lamda)
        loss.append(loss_temp)
    t -= time.time()
    return loss


n,k=X.shape
X=(X-np.mean(X,0))/np.std(X,0)
beta=np.ones((k,1))
loss1=proximal(X,beta,y,0.5,100)
print(loss1)
loss2=acc_proximal1(X,beta,y,0.5,100)
print(loss2)
loss3=acc_proximal2(X,beta,y,0.5,100)
print(loss3)
# X.T.dot(X.dot(beta)-y)

print(np.asarray(loss2)-np.asarray(loss3))



