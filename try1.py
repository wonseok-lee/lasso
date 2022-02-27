from sklearn import datasets
import numpy as np

diabetes = datasets.load_diabetes()
X=diabetes.data
y=diabetes.target.reshape(-1,1)


def soft_threshold_multi(X, lamda):
    '''
    multivariate soft threshold function
    '''
    zero = np.matrix(np.zeros(X.shape))
    return np.multiply(np.sign(X), np.maximum(np.abs(X)-lamda, zero))

def proximal_operator(rho, lamda):
    return soft_threshold_multi(rho, lamda)

def f_x(X, beta, y):
    return 1/2*np.linalg.norm(y-np.matmul(X, beta)) ** 2

def grad_f(X, y, beta):
    XtX_beta = np.matmul(X.T, X).dot(beta)
    Xt_y = np.matmul(X.T, y)
    return XtX_beta-Xt_y

def residual(beta_diff, lamda):
    return 1/(2*lamda)*(np.linalg.norm(beta_diff)**2)

def upper_bound(beta_old,beta_diff, X, y, lamda):
    f_x_val = f_x(X,beta_old,y)
    grad_f_val = grad_f(X,y,beta_old).T.dot(beta_diff)
    residual_val = residual(beta_diff, lamda)
    return f_x_val + grad_f_val + residual_val

def objective_function(X, beta, y, lamda):
    return f_x(X,beta,y)+1/2*lamda*np.sum(np.abs(beta))




def proximal_descent(X,y,beta,lamda,iter_num):
    beta_old=beta
    loss=[]
    t=1
    for j in range(iter_num):
        print(f'j:{j}')
        while True:
            temp=beta_old-lamda*grad_f(X,y,beta_old)
            beta_new=proximal_operator(temp,t*lamda)
            beta_diff=beta_old-beta_new
            if f_x(X,beta_new,y)<upper_bound(beta_old,beta_diff,X,y,lamda):
                break
            else:
                t=0.5*t
        loss.append(objective_function(X,beta_new,y,lamda))

        if j > 1 and np.linalg.norm(objective_function(X,beta_new,y,lamda) - objective_function(X,beta_old,y,lamda)) < 1e-3:
            break
        beta_old=beta_new
    return loss, beta_new

n,k=X.shape
iter_num=100
beta=np.ones((k,1))
lamda=10
loss,beta_new=proximal_descent(X,y,beta,lamda,iter_num)
#
# beta_old=beta
# loss=[]
# t=1
# XtX_beta = np.matmul(X.T, X).dot(beta)
# # temp=beta_old-lamda*grad_f(X,y,beta_old)
# # beta_new=proximal_operator(temp,t*lamda)
# # beta_diff=beta_old-beta_new
# #
# # if f_x(X, beta_new, y) < upper_bound(beta_old, beta_diff, X, y, lamda):
# #     print('good')
# # else:
# #     t = 0.5 * t
# # np.matmul(X.T, X).dot(beta)
# #
# # beta.shape

# XtX_beta = np.matmul(X.T, X).dot(beta)
# Xt_y = np.matmul(X.T, y)