#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#attaching packages 
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
from scipy import stats
import seaborn as sns
import statsmodels
from statsmodels import api as sm
import statsmodels.stats.api as sms
import math
import statsmodels.api as smv


# In[3]:


#loading 5 data sets from home-directory
data = {}
resp = {}
for i in range(1,6):
    fname='X_'+str(i)+'.txt'
    fnamey='Y_'+str(i)+'.txt'
    X=np.loadtxt(fname)
    Y=np.loadtxt(fnamey)
    data[i-1] = X
    resp[i-1] = Y

MINN=np.zeros(5)
MAXX=np.zeros(5)
C=['navy', 'red', 'blue', 'green', 'purple','yellow']
for i in range(0,5):
    MINN[i], MAXX[i] = data[i].min() - 1, data[i].max() + 1
    fig, ax = plt.subplots()
    ax.scatter(data[i], resp[i],color=C[i]);
    ax.legend(['data set_'+str(i+1)])
    


# In[4]:


#distribution of Y
MINNR=np.zeros(5)
MAXXR=np.zeros(5)
for i in range(0,5):
    MINNR[i], MAXXR[i] =resp[i].min(), resp[i].max()
    BINNR= list(np.arange(MINNR[i],MAXXR[i],1))
    fig, ax = plt.subplots()
    pd.Series(resp[i]).plot(kind='hist', bins=BINNR,color=C[i],label="KK")
    plt.text(9, 4.5, 'Y:data_set'+str(i+1))
    
    


# In[5]:


#OLS model
dataP={} 
ols_result={}
for i in range(0,5):
    dataP[i]=sm.add_constant(data[i])
    #print(dataP[i])
    ols_result[i] = sm.OLS(resp[i],dataP[i] ).fit()
    print(ols_result[i].summary())
    #print('Parameters: ', ols_result[i].params)
    #print('R2: ', ols_result[i].rsquared)


# In[6]:


datas={}
Datas={}
for i in range(0,5):
    datas[i]= np.linspace(MINN[i], MAXX[i], 20)
    Datas[i]= sm.add_constant(datas[i])
    fig, ax = plt.subplots()
    ax.scatter(data[i], resp[i],color=C[i]);
    ax.plot(datas[i], ols_result[i].predict(Datas[i]), color=C[i],label='OLS model');
    ax.set_xlim(MINN[i], MAXX[i]);
    ax.legend(loc='upper right');
    ax.legend(['OLS set_'+str(i+1)])


# In[7]:


#test for normality
print("Test For Checking Normality BY Jarque-Bera")
for i in range(0,5):
    name = ['Jarque-Bera', 'Chi^2 two-tail prob.', 'Skew', 'Kurtosis']
    test = sms.jarque_bera(ols_result[i].resid)
    print('data_'+str(i+1), test)

   


# In[8]:


print('Heteroscedasticity Test')
ols_resid={}
resid_fit={}
rho={}
for i in range(0,5):
    ols_resid[i] = ols_result[i].resid
    resid_fit[i] = sm.OLS(ols_resid[i][1:], sm.add_constant(ols_resid[i][:-1])).fit()
    rho[i] = resid_fit[i].params[1]

    #test for Heteroscedasticity Test
for i in range(0,5):
    print('data_set_'+str(i+1),statsmodels.stats.diagnostic.het_white(ols_resid[i]**2, dataP[i], retres=False))


# In[12]:


print('Outliers Identification')
###using Huber's T norm with the default

huber_t={}
hub_results={}
for i in range(0,5):
    huber_t [i]= smv.RLM(resp[i], dataP[i], M=smv.robust.norms.HuberT())
    hub_results[i] = huber_t[i].fit()
    print('min weight for data_set_'+str(i+1),hub_results[i].weights.min())


# In[13]:


#GLS modeling
from scipy.linalg import toeplitz
order={}
sigma={}
gls_model={}
gls_results={}

toeplitz(range(5))
for i in range(0,5):
    order[i] = toeplitz(range(len(ols_resid[i])))
    sigma[i] = rho[i]**order[i]
    gls_model[i] = sm.GLS(resp[i],dataP[i], sigma=sigma[i])
    gls_results[i] = gls_model[i].fit()
    print(gls_results[i].summary())
    


# In[14]:


#Checking ols,gls performance on data without outliers
def mse(actual, predicted):
    return ((actual - predicted)**2).mean()

def RSQ(actual, predicted):
    y_bar=actual.mean()
    re=actual-predicted
    NUMM=sum(np.square(re))
    DOMM=sum(np.square(actual-y_bar))
    RESQ=1-(NUMM/DOMM)
    return(RESQ)

dataP_nout={} 
ols_result_nout={}
data_nout={}
data_nout=data
resp_nout={}
resp_nout=resp

data_nout[0] = data[0][data[0] != 4.0820779]
resp_nout[0] = resp[0][resp[0] != -27.00803487]

data_nout[1] = data[1][data[1] != -8.640359541]
resp_nout[1] = resp[1][resp[1] != 77.64225929]

data_nout[2] = data[2][data[2] != 13.60924464]
resp_nout[2] = resp[2][resp[2] != 78.28113593]


            
for i in range(0,5):
    dataP_nout[i]=sm.add_constant(data_nout[i])
    ols_result_nout[i] = sm.OLS(resp_nout[i],dataP_nout[i] ).fit()
    print(ols_result_nout[i].summary())


Y_ols_nout={}
for i in range (0,5):
    Y_ols_nout[i]=data[i]
    Y_ols_nout[i]=Y_ols_nout[i]*ols_result_nout[i].params[1]+ols_result_nout[i].params[0]
    print(ols_result_nout[i])
    print(RSQ(resp[i],Y_ols_nout[i]))


ols_resid_nout={}
resid_fit_nout={}
rho_nout={}
for i in range(0,5):
    ols_resid_nout[i] = ols_result_nout[i].resid
    resid_fit_nout[i] = sm.OLS(ols_resid_nout[i][1:], sm.add_constant(ols_resid_nout[i][:-1])).fit()
    rho_nout[i] = resid_fit_nout[i].params[1]

from scipy.linalg import toeplitz
order_nout={}
sigma_nout={}
gls_model_nout={}
gls_results_nout={}

toeplitz(range(5))
for i in range(0,5):
    order_nout[i] = toeplitz(range(len(ols_resid_nout[i])))
    sigma_nout[i] = rho_nout[i]**order_nout[i]
    gls_model_nout[i] = sm.GLS(resp_nout[i],dataP_nout[i], sigma=sigma_nout[i])
    gls_results_nout[i] = gls_model_nout[i].fit()
    print(gls_results_nout[i].summary())
    #print(gls_results[i].params)
    
Y_gls_nout={}
for i in range (0,5):
    Y_gls_nout[i]=data[i]
    Y_gls_nout[i]=Y_gls_nout[i]*gls_results_nout[i].params[1]+gls_results_nout[i].params[0]
    print(gls_results_nout[i])
    print(RSQ(resp[i],Y_gls_nout[i]))


# In[15]:


#MLE with a noise dependent to x
from scipy.optimize import minimize
import math
def myfunc(params):
    # print(params)  # <-- you'll see that params is a NumPy array
    a, b, s0, s1 = params # <-- for readability you may wish to assign names to the component variables
    X=np.loadtxt('X_5.txt')
    Y=np.loadtxt('Y_5.txt')
    ss=np.ones((50))
    ress=np.zeros((50))
    kapak=np.zeros((50))
    #print (ress)
    for i in range(0,50):
        ss[i]=((X[i]*s1)**2)+(s0**2)
        ress[i]=((Y[i]-b*X[i]-a)**2)/(2*ss[i])
        #print(ss)
        kapak[i]=math.log(ss[i])
       
    NLL= 0.5*np.sum(kapak)+ np.sum(ress)
    return NLL


myresult= minimize(myfunc, [1 ,1 ,1 ,1],method='BFGS')
print(myresult)


# In[19]:



MLE_coef=([0.05307253, 0.97987193], [-0.43664784,  1.24209258], [-0.03350694, -0.40562675], [8.48970404e-01, 6.65350423e-01],[0.23780701, -0.85762322])
#print(MLE_coef[0][0])
Y_MLX={}
for i in range (0,5):
    Y_MLX[i]=data[i]
    Y_MLX[i]=Y_MLX[i]*MLE_coef[i][1]+MLE_coef[i][0]
    print('MLE coef for data_set_'+str(i+1),MLE_coef[i])
    print('MLE Rsq data_set_'+str(i+1),(RSQ(resp[i],Y_MLX[i])))
    
datas={}
Datas={}
for i in range(0,5):
    datas[i]= np.linspace(MINN[i], MAXX[i], 20)
    Datas[i]= sm.add_constant(datas[i])
    fig, ax = plt.subplots()
    ax.scatter(data[i], resp[i], color=C[i]);
    ax.plot(datas[i], datas[i]*MLE_coef[i][1]+MLE_coef[i][0], color=C[i], label='MLE model');
    ax.set_xlim(MINN[i], MAXX[i]);
    ax.legend(loc='upper right');
    ax.legend(['MLE data set_'+str(i+1)])


# In[17]:


#Baysian Modelling
basic_model={}
y_hat={}
y_obser={}
step={}
trace={}
trace_={}
for i in range(0,5):
    basic_model[i]= pm.Model()
    # Regression coefficients
    with basic_model[i]:
        alpha = pm.Uniform('alpha', -100, 100)
        beta = pm.Uniform('beta', -100, 100)
    
    # Expected value
        y_hat[i] =  data[i]* beta+ alpha 
    # Observations with t-distributed error
        y_obser[i] = pm.StudentT('y_obs[i]', nu=5, mu=y_hat[i], observed=data[i])
        step[i] = pm.NUTS()
        trace_[i] = pm.sample(3000, step[i])
        burn = 1000
        thin = 2
        trace[i] = trace_[i][burn::thin]
        pm.plots.traceplot(trace[i]);


# In[23]:


alpha={}
beta={}
for i in range (0,5):
    alpha[i] = trace[i]['alpha'].mean()
    beta[i] = trace[i]['beta'].mean()
    
Y_Rob={}
for i in range (0,5):
    Y_Rob[i]=data[i]
    Y_Rob[i]=Y_Rob[i]*beta[i]+alpha[i]
    print('Student-T coef for data_set_'+str(i+1),beta[i],alpha[i])
    print('Rsq for data_set_'+str(i+1),(RSQ(resp[i],Y_Rob[i])))
    
datas={}
Datas={}
for i in range(0,5):
    datas[i]= np.linspace(MINN[i], MAXX[i], 20)
    Datas[i]= sm.add_constant(datas[i])
    fig, ax = plt.subplots()
    ax.scatter(data[i], resp[i],color=C[i]);
    ax.plot(datas[i], datas[i]*beta[i]+alpha[i], color=C[i], label='Robust T-dist model');
    ax.set_xlim(MINN[i], MAXX[i]);
    ax.legend(loc='upper right');
    ax.legend(['Stundent T data set_'+str(i+1)])


# In[25]:


#generating Half-Cauchy
modell=pm.Model()
with modell: # model specifications in PyMC3 are wrapped in a with-statement
    # Define priors
    sigma = pm.distributions.continuous.HalfCauchy('sigma', beta=10, testval=1.)
    intercept = pm.distributions.continuous.Normal('Intercept', 0, sd=20)
    x_coeff = pm.distributions.continuous.Normal('data[4]', 0, sd=20)

    # Define likelihood
    likelihood = pm.distributions.continuous.Normal('resp[4]', mu=intercept + x_coeff * data[4],
                        sd=sigma, observed=resp[4])

    # Inference!
    trace = pm.sample(3000, cores=2) # draw 3000 posterior samples using NUTS sampling


# In[26]:


plt.figure(figsize=(7, 7))
pm.traceplot(trace[100:])
plt.tight_layout();
# with modell:
#     # specify glm and pass in data. The resulting linear model, its likelihood and
#     # and all its parameters are automatically added to our model.
#     glm.GLM.from_formula('y ~ x', data[4])
#     trace = pm.sample(3000, cores=2)


# In[27]:


alpha = trace['Intercept'].mean()
beta = trace['data[4]'].mean()
Y_Rob=data[4]
Y_Rob=Y_Rob*beta+alpha
print('coef for data_set_4',alpha,beta)
print('Rsqr', RSQ(resp[4],Y_Rob))
datas= np.linspace(MINN[4], MAXX[4], 20)
Datas= sm.add_constant(datas)
fig, ax = plt.subplots()
ax.scatter(data[4], resp[4]);
ax.plot(datas, datas*beta+alpha, label='data_set 4, Half-Cauchy model');
ax.set_xlim(MINN[4], MAXX[4]);
ax.legend(loc='upper right');


# In[28]:


datas={}
Datas={}
alphaa=[0.0119,-1.3971,2.3556,1.1481,-0.2413]
DD=['red', 'blue', 'purple','gray','orange']
betaa=[1.0538,1.4158,-0.7422,0.6334,-1.3331]
for i in range(0,5):
    datas[i]= np.linspace(MINN[i], MAXX[i], 20)
    Datas[i]= sm.add_constant(datas[i])
    fig, ax = plt.subplots()
    ax.scatter(data[i], resp[i],color=C[i]);
    ax.plot(datas[i], ols_result[i].predict(Datas[i]), color=C[i],label='OLS set_'+str(i+1));
    ax.set_xlim(MINN[i], MAXX[i]);
    
    
    ax.plot(datas[i], alphaa[i] + betaa[i] * datas[i],color=DD[i], label='Best model set_'+str(i+1));

    ax.legend(loc='upper right');


# In[ ]:





# In[ ]:




