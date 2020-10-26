import numpy as np



def derivative_mean(x,mean,rho):
    s=np.log(1+np.exp(rho))
    return (x-mean)/(1.0*s**2)
def derivative_rho(x,mean,rho):
    s=np.log(1+np.exp(rho))
    val1=(x-mean)**2/(1.0*s**3)
    val2=-1/(s*1.0)
    return (val1+val2)*np.exp(rho)/(1+np.exp(rho))