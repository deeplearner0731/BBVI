import time
import numpy as np 
from derivative import derivative_mean, derivative_rho
from data import load_data
from sklearn.preprocessing import scale 
from matplotlib import pyplot as plt




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


def initialization(c,nodes):
    global X,Y,X_test,Y_test
    global kn,p
    X, Y, X_test, Y_test = load_data(seed=0,state=c)
    X_til=np.concatenate((X,X_test))
    X_til=scale(X_til,with_mean=True,with_std=True)
    X=X_til[0:X.shape[0],:]
    X_test=X_til[X.shape[0]:X.shape[0]+X_test.shape[0],:]
    kn=nodes
    p=X.shape[1]

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
    z=np.array(list(map(lambda x:lik_hood(X,Y,Beta[:,x],Gamma[:,:,x]),range(Beta.shape[1]))))
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


def training(combination,tol_diff,tol_chain,itera,cc,num_nodes,fix_learning_rate):
    para={}
    control_theta=False
    parameter=['beta','gamma','all']
    for j in combination:
        print('case: %s_%s_%s'%(j[1],j[0],j[2]))
        
        c=cc
        nodes=num_nodes
        epoch=itera
        lr=fix_learning_rate
        initialization(c,nodes)
        vr_beta_mean=np.zeros(((kn+1),))
        vr_gamma_mean=np.zeros((kn,(p+1)))
        vr_beta_rho=np.ones(((kn+1),))
        vr_gamma_rho=np.ones((kn,(p+1)))
        
        
        chain_len=0
        chain_val=0
        AL,CA1,CA2=[],[],[]
        
        

        for k in parameter:
            para['average_%s_%s_%s_%s'%(k,j[1],j[0],j[2])]=np.zeros((epoch,))
            para['average_%s_derivative_%s_%s_%s'%(k,j[1],j[0],j[2])]=np.zeros((epoch,))


        t0=time.time()
        for step in range(epoch):
            if j[2]=='fix':
                al,vr_beta_mean,vr_beta_rho,vr_gamma_mean,vr_gamma_rho,beta_,gamma_,A,B,C,D=sample(j[1],int(j[0]),lr,j[2],vr_beta_mean,vr_beta_rho,vr_gamma_mean,vr_gamma_rho)
            else:
                al,vr_beta_mean,vr_beta_rho,vr_gamma_mean,vr_gamma_rho,beta_,gamma_,A,B,C,D=sample(j[1],int(j[0]),step,j[2],vr_beta_mean,vr_beta_rho,vr_gamma_mean,vr_gamma_rho)

            para['average_beta_%s_%s_%s'%(j[1],j[0],j[2])][step]=np.mean(np.std(beta_,axis=1))
            para['average_gamma_%s_%s_%s'%(j[1],j[0],j[2])][step]=np.mean(np.std(gamma_.reshape(-1,int(j[0])),axis=1))
            all_=np.concatenate((gamma_.reshape(-1,int(j[0])),beta_),axis=0)
            para['average_all_%s_%s_%s'%(j[1],j[0],j[2])][step]=np.mean(np.std(all_,axis=1))


            para['average_beta_derivative_%s_%s_%s'%(j[1],j[0],j[2])][step]=np.mean(np.std(A,axis=1))
            para['average_gamma_derivative_%s_%s_%s'%(j[1],j[0],j[2])][step]=np.mean(np.std(C.reshape(-1,int(j[0])),axis=1))
            all_=np.concatenate((C.reshape(-1,int(j[0])),A),axis=0)
            para['average_all_derivative_%s_%s_%s'%(j[1],j[0],j[2])][step]=np.mean(np.std(all_,axis=1))
            
            ca1=train_accuracy(False,vr_beta_mean,vr_beta_rho,vr_gamma_mean,vr_gamma_rho)
            ca2=test_accuracy(False,vr_beta_mean,vr_beta_rho,vr_gamma_mean,vr_gamma_rho)
    

            AL=np.append(AL,al)
            CA1=np.append(CA1,ca1)
            CA2=np.append(CA2,ca2)

            if (int(abs(AL[step]))==chain_val):
                chain_len=chain_len+1
            else:
                chain_val=int(abs(AL[step]))
                chain_len=0

            if (step<1):
                diff=abs(AL[step])
            else:
                diff=abs(AL[step]-AL[(step-1)]) 
   
            if ( chain_len>tol_chain): 
                break  

            if ((step+1)%1000==0):
                print(step,al,diff,ca1,ca2)
        tn=time.time()-t0
        para['time_%s_%s_%s'%(j[1],j[0],j[2])]=tn
        para['ELBO_%s_%s_%s'%(j[1],j[0],j[2])]=AL
        para['Testing_%s_%s_%s'%(j[1],j[0],j[2])]=CA2
        para['Traning_%s_%s_%s'%(j[1],j[0],j[2])]=CA1
        
        
        

    return para



def plot_elbo(elbo):

    params = {'legend.fontsize': 'x-large',
              'figure.figsize': (10, 10),
             'axes.labelsize': 'x-large',
             'axes.titlesize':'x-large',
             'xtick.labelsize':'x-large',
             'ytick.labelsize':'x-large'}
    plt.rcParams.update(params)

    plt.plot(elbo,linewidth=5.0)
    plt.title('ELBO')
    plt.xlabel('Iteration')
    plt.ylabel('ELBO')
    plt.legend()
    plt.show()

