from sklearn import datasets
import numpy as np

diabetes = datasets.load_diabetes()
X=diabetes.data
y=diabetes.target.reshape(-1,1)

def f_x(X,beta,y):
    return 1/2*(np.linalg.norm(y-np.matmul(X, beta)) ** 2)

def objective_function(X, beta, y, lamda):
    return f_x(X,beta,y)+lamda*np.sum(np.abs(beta))

def grad_f(X, y, beta):
    XtX_beta = np.matmul(np.matmul(X.T, X),beta)
    Xt_y = np.matmul(X.T, y)
    return XtX_beta-Xt_y

def upperbound(X,y,beta_new,beta_old,lamda):
    return f_x(X,beta_new,y) <= f_x(X,beta_old,y)+np.matmul(grad_f(X,y,beta_old).T,(beta_new-beta_old))+1/(2*lamda)*(np.linalg.norm(beta_new-beta_old)**2)

def soft_threshold_multi(X, rho):
    '''
    multivariate soft threshold function
    '''
    zero = np.matrix(np.zeros(X.shape))
    return np.multiply(np.sign(X), np.maximum(np.abs(X)-rho, zero))

def f_admm(X,y,lr,z,u):
    n,k=X.shape
    return np.linalg.inv(np.eye(k)+1/lr*np.matmul(X.T, X)).dot(z-u+1/lr*np.matmul(X.T,y))

def proximal_descent(X, y, beta, lamda, iter_num, lr):
    beta_old = beta
    loss = []
    r = 0.5

    for j in range(iter_num):
        while True:
            beta_new = soft_threshold_multi(beta_old - lr * grad_f(X, y, beta_old), lr * lamda)
            if upperbound(X, y, beta_new, beta_old, 1):
                break
            else:
                lr = r * lr
        obj_val = objective_function(X, beta_new, y, lamda)

        if j>1 and np.linalg.norm(objective_function(X,beta_new,y,lamda)-objective_function(X,beta_old,y,lamda)) < 1e-4:
            break

        loss.append(obj_val)
        beta_old = beta_new
    return loss, beta_old

def acc_proximal_descent(X, y, beta, lamda, iter_num, lr):
    beta_old = beta
    loss = []
    r = 0.5
    beta_new=beta_old

    for j in range(iter_num):
        if j > 2:
            beta_old = beta_old + (j - 2) / (j + 1) * (beta_new - beta_old)
        while True:
            beta_new = soft_threshold_multi(beta_old - lr * grad_f(X, y, beta_old), lr * lamda)
            if upperbound(X, y, beta_new, beta_old, 1):
                break
            else:
                lr = r * lr
        obj_val = objective_function(X, beta_new, y, lamda)

        if j>1 and np.linalg.norm(objective_function(X,beta_new,y,lamda)-objective_function(X,beta_old,y,lamda)) < 1e-4:
            break

        loss.append(obj_val)
        beta_old = beta_new
    return loss, beta_old

def admm(X,y,beta,lamda, iter_num,lr):
    _,k=X.shape
    beta_old = beta
    loss = []
    alpha_old = np.matrix(np.random.randn(k, 1))
    omega_old = np.matrix(np.random.randn(k, 1))

    for j in range(iter_num):
        beta_new = np.linalg.inv(np.eye(k)+lr*np.matmul(X.T, X)).dot(lr*(alpha_old-omega_old)+np.matmul(X.T,y))
        # f_admm(X, y, lr, alpha_old, omega_old)
        alpha_new=soft_threshold_multi(beta_new+omega_old,lamda*lr)
        omega_new=omega_old+beta_new-alpha_old

        obj_val = objective_function(X, beta_new, y, lamda)
        loss.append(obj_val)

        if j>1 and np.linalg.norm(objective_function(X,beta_new,y,lamda)-objective_function(X,beta_old,y,lamda)) < 1e-4:
            break

        beta_old = beta_new
        alpha_old = alpha_new
        omega_old = omega_new

    return loss, beta_old

X=(X-np.mean(X,0))/np.std(X,0)
n,k=X.shape
beta=np.ones((k,1))
lamda=10
iter_num=100
loss=[]
lr=1e-2
r=0.5
loss_proximal, _ = proximal_descent(X, y, beta, lamda, iter_num, lr)
loss_acc_proximal, _ = acc_proximal_descent(X, y, beta, lamda, iter_num, lr)
loss_admm, _ = admm(X, y, beta, lamda, iter_num, lr)

print(len(loss_proximal))
print(len(loss_acc_proximal))
print(len(loss_admm))

print(loss_proximal)
print(loss_admm)
