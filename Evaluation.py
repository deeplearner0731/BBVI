import numpy as np

def q_simulate(no_sample,vr_beta_mean,vr_beta_rho,vr_gamma_mean,vr_gamma_rho):
    Beta=np.zeros(((kn+1),no_sample))
    for i in range((kn+1)):
        Beta[i,:]=np.random.normal(vr_beta_mean[i], np.log(1+np.exp(vr_beta_rho[i])), no_sample)
    Gamma=np.zeros((kn,(p+1),no_sample))
    for i in range(kn):
        for j in range((p+1)):
             Gamma[i,j,:]=np.random.normal(vr_gamma_mean[i][j], np.log(1+np.exp(vr_gamma_rho[i][j])), no_sample)    
    return Beta,Gamma


def train_accuracy(theta_var,vr_beta_mean,vr_beta_rho,vr_gamma_mean,vr_gamma_rho):
    Beta,Gamma=q_simulate(10000,vr_beta_mean,vr_beta_rho,vr_gamma_mean,vr_gamma_rho)
    beta=np.apply_along_axis(np.mean, 1, Beta)
    gamma=np.apply_along_axis(np.mean, 2, Gamma)
    n=X.shape[0]
    X1=np.column_stack((np.ones(n),X))
    V1=np.dot(gamma,X1.T)
    V2=1.0/(1+np.exp(-V1))
    theta=beta[0]+np.dot(beta[1::].reshape(1,kn),V2)
    if theta_var==True:
        var_theta=np.std(theta,axis=1)
    
        Y_est=np.array([0 if x<0.5 else 1 for x in 1.0/(1+np.exp(-theta.T))])
        Y_est=Y_est.reshape((len(Y_est),))
        acc=sum(Y==Y_est)*1.0/len(Y)
        return acc,var_theta
    else:
        Y_est=np.array([0 if x<0.5 else 1 for x in 1.0/(1+np.exp(-theta.T))])
        Y_est=Y_est.reshape((len(Y_est),))
        acc=sum(Y==Y_est)*1.0/len(Y)
        return acc
        

def test_accuracy(theta_var,vr_beta_mean,vr_beta_rho,vr_gamma_mean,vr_gamma_rho):
    Beta,Gamma=q_simulate(10000,vr_beta_mean,vr_beta_rho,vr_gamma_mean,vr_gamma_rho)
    beta=np.apply_along_axis(np.mean, 1, Beta)
    gamma=np.apply_along_axis(np.mean, 2, Gamma)
    n=X_test.shape[0]
    X1=np.column_stack((np.ones(n),X_test))
    V1=np.dot(gamma,X1.T)
    V2=1.0/(1+np.exp(-V1))
    theta=beta[0]+np.dot(beta[1::].reshape(1,kn),V2)
    if theta_var==True:
        var_theta=np.std(theta,axis=1)
        Y_est=np.array([0 if x<0.5 else 1 for x in 1.0/(1+np.exp(-theta.T))])
        Y_est=Y_est.reshape((len(Y_est),))
        acc=sum(Y_test==Y_est)*1.0/len(Y_test)
        return acc, var_theta
    else:
        Y_est=np.array([0 if x<0.5 else 1 for x in 1.0/(1+np.exp(-theta.T))])
        Y_est=Y_est.reshape((len(Y_est),))
        acc=sum(Y_test==Y_est)*1.0/len(Y_test)
        return acc