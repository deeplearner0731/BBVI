import numpy as np 


def gaussian_distribution(x,mu,rho):
    sigma=np.log(1+np.exp(rho))
    return (1/sigma)*np.exp(-1/(2.0*sigma**2)*(x-mu)**2)


def cov(x,y):
    cov_bias = np.mean(x * y, axis=1) - np.mean(x, axis=1) * np.mean(y, axis=1)
    return cov_bias

def q_simulate(no_sample,vr_beta_mean,vr_beta_rho,vr_gamma_mean,vr_gamma_rho):
    Beta=np.zeros(((kn+1),no_sample))
    for i in range((kn+1)):
        Beta[i,:]=np.random.normal(vr_beta_mean[i], np.log(1+np.exp(vr_beta_rho[i])), no_sample)
    Gamma=np.zeros((kn,(p+1),no_sample))
    for i in range(kn):
        for j in range((p+1)):
             Gamma[i,j,:]=np.random.normal(vr_gamma_mean[i][j], np.log(1+np.exp(vr_gamma_rho[i][j])), no_sample)    
    return Beta,Gamma


def log_p_q(Beta,Gamma,Vr_beta_mean,Vr_beta_rho,Vr_gamma_mean,Vr_gamma_rho):
    z=np.array(list(map(lambda x:data_(X,Y,Beta[:,x],Gamma[:,:,x]),range(Beta.shape[1]))))
    B1=(np.log(gaussian_distribution(Beta,Vr_beta_mean,Vr_beta_rho))).T
    b1=np.sum(B1,1)
    B2=(np.log(gaussian_distribution(Beta,0,np.log(1+np.exp(1))))).T
    b2=np.sum(B2,1)
    G1=(np.log(gaussian_distribution(Gamma,Vr_gamma_mean,Vr_gamma_rho))).transpose(2,0,1)
    g1=np.sum(G1,(1,2))
    G2=(np.log(gaussian_distribution(Gamma,0,np.log(1+np.exp(1))))).transpose(2,0,1)
    g2=np.sum(G2,(1,2))
    prior_value=z+b2+g2
    posterior_value=b1+g1
    return prior_value-posterior_value


def lik_hood(x,y,beta,gamma):
    n=x.shape[0]
    x=np.column_stack((np.ones(n),x))
    val1=np.dot(gamma,x.T)
    val2=1.0/(1+np.exp(-val1))
    theta=beta[0]+np.dot(beta[1::].reshape(1,kn),val2)
    return np.sum(y*theta-np.log(1+np.exp(theta)))



def sample(type_,no_sample,lr,fix,vr_beta_mean,vr_beta_rho,vr_gamma_mean,vr_gamma_rho):
    Beta,Gamma=q_simulate(no_sample,vr_beta_mean,vr_beta_rho,vr_gamma_mean,vr_gamma_rho)
    Vr_beta_mean=np.outer(vr_beta_mean,np.ones((no_sample,)))
    Vr_beta_rho=np.outer(vr_beta_rho,np.ones((no_sample,)))
    Vr_gamma_mean=np.outer(vr_gamma_mean,np.ones((no_sample,))).reshape((kn,(p+1),no_sample))
    Vr_gamma_rho=np.outer(vr_gamma_rho,np.ones((no_sample,))).reshape((kn,(p+1),no_sample))
    L=log_p_q(Beta,Gamma,Vr_beta_mean,Vr_beta_rho,Vr_gamma_mean,Vr_gamma_rho)
    A=derivative_mean(Beta,Vr_beta_mean,Vr_beta_rho)
    B=derivative_rho(Beta,Vr_beta_mean,Vr_beta_rho)
    C=derivative_mean(Gamma,Vr_gamma_mean,Vr_gamma_rho)
    D=derivative_rho(Gamma,Vr_gamma_mean,Vr_gamma_rho)
    if fix=='variate':
        learning_rate=float(1)/(100*(lr+1)**0.3)
    else:
        learning_rate=lr
        
    if type_=='bbvi_':
        vr_beta_mean=vr_beta_mean+learning_rate*(np.dot(A,L)/no_sample)
        vr_beta_rho=vr_beta_rho+learning_rate*(np.dot(B,L)/no_sample)
        vr_gamma_mean=vr_gamma_mean+learning_rate*(np.dot(C,L)/no_sample)
        vr_gamma_rho=vr_gamma_rho+learning_rate*(np.dot(D,L)/no_sample)
        
    elif type_=='bbvi_cv':
        fd1=A*L
        var1=np.var(A, axis=1)
        cov1=cov(fd1,A)
        a1=cov1/var1
        b1=(A.reshape(no_sample,kn+1)*a1).reshape(kn+1,no_sample)
        val1=np.sum(fd1-b1,axis=1)
        vr_beta_mean=vr_beta_mean+learning_rate*(val1/no_sample)



        fd2=B*L
        var2=np.var(B, axis=1)
        cov2=cov(fd2,B)
        a2=cov2/var2
        b2=(B.reshape(no_sample,kn+1)*a2).reshape(kn+1,no_sample)
        val2=np.sum(fd2-b2,axis=1)
        vr_beta_rho=vr_beta_rho+learning_rate*(val2/no_sample)



    ################################################################

        fd3=C*L
        fd3_reshape=fd3.reshape(kn*(p+1),-1)
        C_reshape=C.reshape(kn*(p+1),-1)
        var3=np.var(C_reshape, axis=1)
        cov3=cov(fd3_reshape,C_reshape)
        a3=cov3/var3
        a3_reshape=a3.reshape(kn*(p+1),-1)
        b3=C_reshape*a3_reshape
        val3=np.sum(fd3_reshape-b3,axis=1)
        val3_reshape=val3.reshape(kn,p+1) 
        vr_gamma_mean=vr_gamma_mean+learning_rate*(val3_reshape/no_sample)




        fd4=D*L
        fd4_reshape=fd4.reshape(kn*(p+1),-1)
        D_reshape=D.reshape(kn*(p+1),-1)
        var4=np.var(D_reshape, axis=1)
        cov4=cov(fd4_reshape,D_reshape)
        a4=cov4/var4
        a4_reshape=a4.reshape(kn*(p+1),-1)
        b4=D_reshape*a4_reshape
        val4=np.sum(fd4_reshape-b4,axis=1)
        val4_reshape=val4.reshape(kn,p+1) 
        vr_gamma_rho=vr_gamma_rho+learning_rate*(val4_reshape/no_sample)

    
    return np.mean(L),vr_beta_mean,vr_beta_rho,vr_gamma_mean,vr_gamma_rho, Beta, Gamma, A,B,C,D