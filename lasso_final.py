import time
from sklearn import datasets
import numpy as np

class Record:
    def __init__(self):
        self.obj = []
        self.loss = []
        self.time = []

    def add_obj(self, val):
        self.obj.append(val)

    def add_loss(self, val):
        self.loss.append(val)

    def add_time(self, val):
        self.time.append(val)


def soft_threshold(X, rho):
    '''
    :param X: Data Matrix
    :param rho: lamda
    :return: soft_threshold
    '''
    zero = np.matrix(np.zeros(X.shape))
    return np.multiply(np.sign(X), np.maximum(np.abs(X)-rho, zero))

def objective_function(X, beta, y, lamda):
    '''
    :param X: Data matrix
    :param beta: beta
    :param y: target value
    :param lamda: penalty value
    :return: 1/2 * ||y-X*beta||_2^2 + lamda * ||beta||_1
    '''
    return 1/2*np.linalg.norm(X.dot(beta)-y)**2+lamda*np.linalg.norm(beta,1)

def coordinate_descent(X, beta, y, lamda, iter_num):
    '''
    coordinate descent for lasso regression (L2 penalized Regression)

    :param X: Data matrix
    :param beta: initial beta
    :param y: target value
    :param lamda: penalty value
    :param iter_num: iteration number to infer beta
    :return: lists including time, objective value, MSE(loss)
    '''
    _, k = X.shape
    record = Record()
    t = time.time()
    for _ in range(iter_num):

        # update beta
        for j in range(k):
            X_j = X[:, j].reshape(-1, 1)
            y_pred = np.matmul(X, beta)
            beta_j = beta[j]
            rho = np.matmul(X_j.T,(y-y_pred+beta_j*X_j))
            beta[j] = soft_threshold(rho, lamda)

        obj_temp = objective_function(X, beta, y, lamda)
        loss_temp = np.linalg.norm(y - X.dot(beta)) ** 2
        record.add_obj(obj_temp)
        record.add_loss(loss_temp)

    t -= time.time()
    record.add_time(t)
    return record

def proximal(X,beta,y,lamda,iter_num):
    '''
    proximal descent for lasso regression (L2 penalized Regression)

    :param X: Data matrix
    :param beta: initial beta
    :param y: target value
    :param lamda: penalty value
    :param iter_num: iteration number to infer beta
    :return: lists including time, objective value, MSE(loss)
    '''
    c = np.linalg.norm(X) ** 2
    record = Record()
    t = time.time()
    for j in range(iter_num):
        temp_beta = beta-lamda*np.matmul(X.T,(X.dot(beta)-y))

        # update beta
        beta_new = soft_threshold(temp_beta/c,lamda/c)

        obj_temp = objective_function(X,beta_new,y,lamda)
        loss_temp = np.linalg.norm(y-X.dot(beta_new)) ** 2
        record.add_obj(obj_temp)
        record.add_loss(loss_temp)
        beta = beta_new
    t -= time.time()
    record.add_time(t)
    return record

def acc_proximal1(X,beta,y,lamda,iter_num):
    '''
    accelerated proximal descent for lasso regression (L2 penalized Regression)
    momentum = (j - 2) / (j + 1) * (beta_j - beta_{j-1})

    :param X: Data matrix
    :param beta: initial beta
    :param y: target value
    :param lamda: penalty value
    :param iter_num: iteration number to infer beta
    :return: lists including time, objective value, MSE(loss)
    '''
    c = np.linalg.norm(X) ** 2
    record = Record()
    t = time.time()
    for j in range(1,iter_num+1):
        if j > 1:
            # update momentum
            v = beta + (j - 2) / (j + 1) * (beta - beta_old)
            temp_beta = v - lamda * np.matmul(X.T, (X.dot(v) - y))

            # update beta
            beta_new = soft_threshold(temp_beta / c, lamda / c)

            obj_temp = objective_function(X, beta_new, y, lamda)
            loss_temp = np.linalg.norm(y - X.dot(beta_new)) ** 2
            record.add_obj(obj_temp)
            record.add_loss(loss_temp)
            beta, beta_old = beta_new, beta
        else:
            temp_beta = beta - lamda * np.matmul(X.T, (X.dot(beta) - y))

            # update beta
            beta = soft_threshold(temp_beta / c, lamda / c)

            obj_temp = objective_function(X, beta_new, y, lamda)
            loss_temp = np.linalg.norm(y - X.dot(beta_new)) ** 2
            record.add_obj(obj_temp)
            record.add_loss(loss_temp)
            beta_old = beta
    t -= time.time()
    record.add_time(t)
    return record

def acc_proximal2(X,beta,y,lamda,iter_num):
    '''
    accelerated proximal descent for lasso regression (L2 penalized Regression)
    momentum = sqrt((1 + 4*(j**2)))/2

    :param X: Data matrix
    :param beta: initial beta
    :param y: target value
    :param lamda: penalty value
    :param iter_num: iteration number to infer beta
    :return: lists including time, objective value, MSE(loss)
    '''
    c = np.linalg.norm(X) ** 2
    k = 1
    v = beta
    record = Record()
    t = time.time()
    for j in range(1,iter_num+1):
        beta_old = beta
        temp_beta = v - lamda * np.matmul(X.T, (X.dot(v) - y))

        # update beta
        beta = soft_threshold(temp_beta / c, lamda / c)

        # update momentum
        k0 = k
        k = (1 + np.sqrt(1 + 4*(k**2)))/2
        v = beta + ((k0 - 1)/k) * (beta - beta_old)

        obj_temp = objective_function(X, beta, y, lamda)
        loss_temp = np.linalg.norm(y - X.dot(beta)) ** 2
        record.add_obj(obj_temp)
        record.add_loss(loss_temp)
    t -= time.time()
    record.add_time(t)
    return record

def admm(X,y,lamda,iter_num):
    '''
    ADMM for lasso regression (L2 penalized Regression)

    :param X: Data matrix
    :param y: target value
    :param lamda: penalty value
    :param iter_num: iteration number to infer beta
    :return: lists including time, objective value, MSE(loss)
    '''
    n, k = X.shape
    alpha = np.ones((k,1))
    omega = np.ones((k, 1))
    c = np.linalg.norm(X) ** 2
    t = time.time()
    record = Record()
    for j in range(iter_num):
        # update beta, alpha, omega
        beta_new = np.linalg.inv(X.T.dot(X)+c*np.eye(k)).dot(X.T.dot(y) + c*(alpha-omega))
        alpha_new = soft_threshold(beta_new+omega,lamda/c)
        omega_new = omega + beta_new - alpha_new

        obj_temp = objective_function(X, beta_new, y, lamda)
        loss_temp = np.linalg.norm(y - X.dot(beta_new)) ** 2
        record.add_obj(obj_temp)
        record.add_loss(loss_temp)

        beta, alpha, omega = beta_new, alpha_new, omega_new
    t -= time.time()
    record.add_time(t)
    return record


diabetes = datasets.load_diabetes()
X=diabetes.data
y=diabetes.target.reshape(-1,1)

n,k=X.shape
X=(X-np.mean(X,0))/np.std(X,0)
beta=np.ones((k,1))
loss1=proximal(X,beta,y,0.5,100)
print(loss1)
# loss2=acc_proximal1(X,beta,y,0.5,100)
# print(loss2)
loss3=admm(X,y,0.5,100)
print(loss3)
# X.T.dot(X.dot(beta)-y)




