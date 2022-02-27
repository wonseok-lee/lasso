import numpy as np

def soft_threshold_uni(x, lamda):
    '''
    univariate soft threshold function
    '''
    if x < -lamda:
        return x + lamda
    elif x > lamda:
        return x - lamda
    else:
        return 0

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

def f_admm(X,y,r,z,u):
    n,k=X.shape
    return np.linalg.inv(np.eye(n)+1/r*np.matmul(X.T, X)).dot(z-u+1/r*np.matmul(X.T,y))

def coordinate_descent(iter_num, X, y, beta, lamda):
    '''
    coordinate descent
    '''
    _, k = X.shape
    loss = []
    for _ in range(iter_num):
        for j in range(k):
            X_j = X[:, j].reshape(-1, 1)
            y_pred = np.matmul(X, beta)
            beta_j = beta[j]
            rho = np.matmul(X_j.T,(y-y_pred+beta_j*X_j))
            beta[j] = soft_threshold_uni(rho, lamda)
        loss.append(objective_function(X,beta,lamda))
    return loss, beta

def proximal_descent(X,y,beta,lamda,iter_num):
    beta_old=beta
    loss=[]
    t=1
    for _ in range(iter_num):
        while True:
            temp=beta_old-lamda*grad_f(X,y,beta_old)
            beta_new=proximal_operator(temp,t*lamda)
            beta_diff=beta_old-beta_new
            if f_x(X,beta_new,y)<upper_bound(beta_old,beta_diff,X,y,lamda):
                break
            else:
                t=0.5*t
            loss.append(objective_function(X,beta_new,lamda))
    return loss, beta_new

def acc_proximal_descent(X, y, beta, lamda, iter_num):
    beta_old = beta
    loss = []
    beta_new = beta_old
    t = 1
    for K in range(iter_num):
        if K > 2:
            beta_old = beta_old + (K - 2) / (K + 1) * (beta_new - beta_old)
        while True:
            temp = beta_old - lamda * grad_f(X, y, beta_old)
            beta_new = proximal_operator(temp, t * lamda)
            beta_diff = beta_old - beta_new
            if f_x(X, beta_new, y)<upper_bound(beta_old, beta_diff, X, y, lamda):
                break
            else:
                t=0.5*t
            loss.append(objective_function(X, beta_new, lamda))
    return loss, beta_new


def admm(X, y, beta, lamda, iter_num, r):
    _,n=X.shape
    beta_old = beta
    loss = []
    beta_new = beta_old
    z_old = np.matrix(np.random.randn(n, 1))
    u_old = np.matrix(np.random.randn(n, 1))
    for K in range(iter_num):
        beta_new = f_admm(X, y, r, z_old, u_old, n)
        temp = beta_new + u_old
        z_new = proximal_operator(temp, lamda/r)
        u_new = u_old + beta_new - z_new

        loss.append(objective_function(X, beta_new, lamda))

        beta_old = beta_new
        u_old = u_new
        z_old = beta_new

    return loss, beta_new

def f_admm(X,y,r,z,u,n):
    return np.linalg.inv(np.eye(n)+r*np.matmul(X.T, X)).dot(r*(z-u)+np.matmul(X.T,y))


