import numpy as np 
import matplotlib.pyplot as plt 

# Class for Linear Discriminant Analysis :
class LinearDiscriminantAnalysis():

    def __init__(self,):
        '''
        Attributes:
        
        mu_0: np.array
            mean of p(x/y=0)
        mu_1: np.array
            mean of p(x/y=1)
        pi: float
            parameter of p(y)   
        sigma: np.array
            covariance matrix of p(x/y)            
        '''
        
        self.mu_0 = None
        self.mu_1 = None
        self.pi = None
        self.sigma = None

    def fit(self, X, Y):
        """ Fit the data (X, Y).
    
        Parameters:
        -----------
        X: (num_samples, num_features) np.array
            Design matrix
        Y: (num_samples, 1) np.array
            Output vector
        Note:
        -----
        Compute parameters
        """
        self.pi = np.mean(Y)
        self.mu_0 = np.sum((1-Y)*X , axis=0)/np.sum(1-Y)
        self.mu_1 = np.sum(Y*X, axis=0)/np.sum(Y)
        self.sigma = (1/X.shape[0])*(np.dot((X-self.mu_1).T,Y*(X-self.mu_1)) + np.dot((X-self.mu_0).T,(1-Y)*(X-self.mu_0)))

        
    def predict(self, X):
        """ Make predictions for data X.
    
        Parameters:
        -----------
        X: (num_samples, num_features) np.array
            New matrix
        
        Returns:
        -----
        y_pred: (num_samples, ) np.array
            Predictions
        """
        b = np.dot(np.linalg.inv(self.sigma).T,(self.mu_1-self.mu_0))
        a = np.log(self.pi/(1-self.pi)) + 0.5*(np.dot(self.mu_0.T,np.dot(np.linalg.inv(self.sigma),self.mu_0))- np.dot(self.mu_1.T,np.dot(np.linalg.inv(self.sigma),self.mu_1)) )
        P =  a + np.dot(X,b)
        P = 1/(1+np.exp(-P))
        return np.where(P <= 0.5,0,1)
    
def plot_contour_LDA(LDA,dataset,x_train,y_train,x_test,y_test,saving=True) :

    x,y,x_test,y_test = x_train,y_train,x_test,y_test
        
    pi , mu_0 , mu_1 , sigma = LDA.pi , LDA.mu_0, LDA.mu_1 , LDA.sigma

    b = np.dot(np.linalg.inv(sigma),(mu_1-mu_0))
    a = np.log(pi/(1-pi)) + 0.5*(np.dot(mu_0.T,np.dot(np.linalg.inv(sigma),mu_0))- np.dot(mu_1.T,np.dot(np.linalg.inv(sigma),mu_1)) )
    
    f, (ax1, ax2) = plt.subplots(1,2, figsize=(14,5))
    
    # Drawing the contour p(y=1/x)=0.5
    x_min, x_max = x[:,0].min() - 1,x[:,0].max() + 1
    y_min, y_max = x[:,1].min() - 1,x[:,1].max() + 1
    X, Y = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))
    F= b[1]*Y +b[0]*X + a
    ax1.contour(X,Y,F,[0],colors="black")
    ax1.contourf(X, Y, F, 0, alpha=0.5, colors=["darksalmon","lightblue"])
    # Drawing point for test set
    ax1.scatter(x[y.reshape(-1)==0,0],x[y.reshape(-1)==0,1],c="saddlebrown",label="y=0")
    ax1.scatter(x[y.reshape(-1)==1,0],x[y.reshape(-1)==1,1],c="navy",label="y=1")
    ax1.set_title('LDA on train data '+dataset)
    ax1.legend()
    

    # Drawing the contour p(y=1/x)=0.5
    x_min, x_max = x_test[:,0].min() - 1,x_test[:,0].max() + 1
    y_min, y_max = x_test[:,1].min() - 1, x_test[:,1].max() + 1
    X, Y = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))
    F= b[1]*Y +b[0]*X + a
    ax2.contour(X,Y,F,[0],colors="black")
    ax2.contourf(X, Y, F, 0, alpha=0.5, colors=["darksalmon","lightblue"])
    # Drawing point for test set
    ax2.scatter(x_test[y_test.reshape(-1)==0,0],x_test[y_test.reshape(-1)==0,1],c="saddlebrown",label="y=0")
    ax2.scatter(x_test[y_test.reshape(-1)==1,0],x_test[y_test.reshape(-1)==1,1],c="navy",label="y=1")
    ax2.set_title('LDA on test data '+dataset)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    y_pred_train=LDA.predict(x).flatten()
    acc_train=(np.sum(y_pred_train==y.flatten())/y.shape[0])*100
    print("\t LDA Train Accuracy :",np.round(acc_train,2),end="\t\t\t\t\t ")
    
    y_pred_test=LDA.predict(x_test).flatten()
    acc_test=(np.sum(y_pred_test==y_test.flatten())/y_test.shape[0])*100
    print("LDA Test Accuracy :",np.round(acc_test,2),"\n")

    print("\t LDA Train Misclassification error:",np.round(100-acc_train,2),end="\t\t")
    print("\t LDA Test Misclassification error:",np.round(100-acc_test,2),end="\t\t")
    
    # Saving Figures
    if saving:
        import os
        if not os.path.isdir("images/"): 
            os.makedirs("images/")   
        extent = ax1.get_window_extent().transformed(f.dpi_scale_trans.inverted())
        f.savefig('images/LDA_'+dataset+'.png', bbox_inches=extent.expanded(1.075, 1.2))
        extent = ax2.get_window_extent().transformed(f.dpi_scale_trans.inverted())
        f.savefig('images/LDA_test_'+dataset+'.png', bbox_inches=extent.expanded(1.1, 1.2))
        
        
        
# Class for Logistic Regression :
class LogisticRegression():

    def __init__(self,max_iter=100, epsilon=1e-3):
        '''
        Attributes:
        
        weights: np.array
            weights vector
        grad: np.array
            Loss gradient vector
        hess: np.array
            Loss Hessian matrix 
        max_iter: np.array
            maximum iterations for the IRLS algorithm 
        epsilon: float
            regularization coefficient for the IRLS algorithm
        '''
        self.weights = None
        self.grad = None
        self.hess = None
        self.nu = None
        self.max_iter_ = max_iter
        self.epsilon_ = epsilon
        
    def fit(self, X, Y):
        """ Fit the data (X, Y).
    
        Parameters:
        -----------
        X: (num_samples, num_features) np.array
            Design matrix
        Y: (num_sampes, ) np.array
            Output vector
        Note:
        -----
        Compute weights through IRLS algorithm
        """
        X_aug = np.hstack((np.ones((X.shape[0],1)),X))
        n,p=X_aug.shape
        self.weights = np.random.RandomState(1).normal(0,0.5,(p,1))
        for i in range(self.max_iter_):
            self.nu = 1/(1 + np.exp(-np.dot(X_aug,self.weights)))
            self.grad = np.dot(X_aug.T,(Y-self.nu))
            self.hess = -np.dot( np.dot(X_aug.T, np.diag((self.nu*(1-self.nu))[:,0])) , X_aug ) + self.epsilon_*np.eye(p)
            self.weights = self.weights - np.dot(np.linalg.inv(self.hess),self.grad)
            # Other method :
            #self.weights = np.linalg.solve(self.hess,np.dot(self.hess,self.weights)-self.grad)
            
            
    def predict(self, X):
        """ Make predictions for data X.
    
        Parameters:
        -----------
        X: (num_samples, num_features) np.array
            New matrix
        
        Returns:
        -----
        y_pred: (num_samples, ) np.array
            Predictions
        """
        X_aug = np.hstack((np.ones((X.shape[0],1)),X))
        P = 1/(1+np.exp(-np.dot(X_aug,self.weights)))
        return np.where(P<=0.5,0,1)
    

def plot_contour_LR(LR,dataset,x_train,y_train,x_test,y_test,saving=True) : 
    #if dataset == 'A' : 
    #    x,y,x_test,y_test = x_train_A,y_train_A,x_test_A,y_test_A
    #elif dataset == 'B' :
    #    x,y,x_test,y_test = x_train_B,y_train_B,x_test_B,y_test_B
    #else : 
    #    x,y,x_test,y_test = x_train_C,y_train_C,x_test_C,y_test_C
    
    x,y,x_test,y_test = x_train,y_train,x_test,y_test
    
    w = LR.weights
    
    f, (ax1, ax2) = plt.subplots(1,2, figsize=(14,5))
    
    # Drawing the contour p(y=1/x)=0.5
    x_min, x_max = x[:,0].min() - 1,x[:,0].max() + 1
    y_min, y_max = x[:,1].min() - 1,x[:,1].max() + 1
    X, Y = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))
    F= w[2]*Y +w[1]*X + w[0]
    ax1.contour(X,Y,F,[0],colors="black")
    ax1.contourf(X, Y, F, 0, alpha=0.5, colors=["darksalmon","lightblue"])
    # Drawing point for test set
    ax1.scatter(x[y.reshape(-1)==0,0],x[y.reshape(-1)==0,1],c="saddlebrown",label="y=0")
    ax1.scatter(x[y.reshape(-1)==1,0],x[y.reshape(-1)==1,1],c="navy",label="y=1")
    ax1.set_title('Logistic Regression on train data '+dataset)
    ax1.legend()
    

    # Drawing the contour p(y=1/x)=0.5
    x_min, x_max = x_test[:,0].min() - 1,x_test[:,0].max() + 1
    y_min, y_max = x_test[:,1].min() - 1, x_test[:,1].max() + 1
    X, Y = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))
    F= w[2]*Y +w[1]*X + w[0]
    ax2.contour(X,Y,F,[0],colors="black")
    ax2.contourf(X, Y, F, 0, alpha=0.5, colors=["darksalmon","lightblue"])
    # Drawing point for test set
    ax2.scatter(x_test[y_test.reshape(-1)==0,0],x_test[y_test.reshape(-1)==0,1],c="saddlebrown",label="y=0")
    ax2.scatter(x_test[y_test.reshape(-1)==1,0],x_test[y_test.reshape(-1)==1,1],c="navy",label="y=1")
    ax2.set_title('Logistic Regression on test data '+dataset)
    ax2.legend()
    
    plt.tight_layout()
    plt.show() 
    
    y_pred_train=LR.predict(x).flatten()
    acc_train=(np.sum(y_pred_train==y.flatten())/y.shape[0])*100
    print("\t LogReg Train Accuracy :",np.round(acc_train,2),end="\t\t\t\t\t ")
    
    y_pred_test=LR.predict(x_test).flatten()
    acc_test=(np.sum(y_pred_test==y_test.flatten())/y_test.shape[0])*100
    print("LogReg Test Accuracy :",np.round(acc_test,2),"\n")
    
    print("\t LogReg Train Misclassification error:",np.round(100-acc_train,2),end="\t\t")
    print("\t LogReg Test Misclassification error:",np.round(100-acc_test,2),end="\t\t")
    
    # Saving Figures
    if saving:
        import os
        if not os.path.isdir("images/"): 
            os.makedirs("images/") 
        extent = ax1.get_window_extent().transformed(f.dpi_scale_trans.inverted())
        f.savefig('images/Logistic_'+dataset+'.png', bbox_inches=extent.expanded(1.075, 1.2))
        extent = ax2.get_window_extent().transformed(f.dpi_scale_trans.inverted())
        f.savefig('images/Logistic_test'+dataset+'.png', bbox_inches=extent.expanded(1.1, 1.2))
        

# Class for Linear Regression :
class LinearRegression():
    
    def __init__(self, alpha=0):
        '''
        Attributes:
        
        alpha: float
            ridge regularization coefficient
        beta: np.array
            weights vector
        sigma: float
            standard deviation of p(y/x)
        '''
        
        self.alpha = alpha
        self.beta = None
        self.sigma = None

                
    def fit(self, X, y):
        """         
        Parameters:
        -----------
        X: (n, p) np.array
            Data matrix
        y: (n, 1) np.array
            Outputs array
        Returns:
        -----
        Compute parameters and weights
        """
        X_aug=np.hstack((np.ones((X.shape[0],1)),X))
        n,p=X_aug.shape
        self.beta = np.dot(np.matmul(np.linalg.inv(np.matmul(X_aug.T,X_aug)+self.alpha*np.eye(p)),X_aug.T),y)
        self.sigma = np.mean((y-np.matmul(X_aug,self.beta))**2)
        
        
    def predict(self, X):
        """ Predict output for X
        
        Parameters:
        -----------
        X: (n, p) np.array
            New data matrix
        
        Returns:
        -----
        Outputs        
        """
        X_aug=np.hstack((np.ones((X.shape[0],1)),X))
        return np.where(np.dot(X_aug,self.beta)<=0.5,0,1)
    

def plot_contour_LinR(LinReg,dataset,x_train,y_train,x_test,y_test,saving=True):
    #if dataset == 'A' : 
    #    x,y,x_test,y_test = x_train_A,y_train_A,x_test_A,y_test_A
    #elif dataset == 'B' :
    #    x,y,x_test,y_test = x_train_B,y_train_B,x_test_B,y_test_B
    #else : 
    #    x,y,x_test,y_test = x_train_C,y_train_C,x_test_C,y_test_C
        
    x,y,x_test,y_test = x_train,y_train,x_test,y_test
    
    f, (ax1,ax2) = plt.subplots(1,2, figsize=(14,5))
    
    # Drawing the contour p(y=1/x)=0.5
    x_min, x_max = x[:,0].min() - 1,x[:,0].max() + 1
    y_min, y_max = x[:,1].min() - 1,x[:,1].max() + 1
    X, Y = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))
    F=LinReg.beta[2]*Y + LinReg.beta[1]*X + LinReg.beta[0] - 0.5
    ax1.contour(X,Y,F,[0],colors="black")
    ax1.contourf(X, Y, F, 0, alpha=0.5, colors=["darksalmon","lightblue"])
    # Drawing point for test set
    ax1.scatter(x[y.reshape(-1)==0,0],x[y.reshape(-1)==0,1],c="saddlebrown",label="y=0")
    ax1.scatter(x[y.reshape(-1)==1,0],x[y.reshape(-1)==1,1],c="navy",label="y=1")
    ax1.set_title('Linear Regression on train data '+dataset)
    ax1.legend()

    # Drawing the contour p(y=1/x)=0.5
    x_min, x_max = x_test[:,0].min() - 1,x_test[:,0].max() + 1
    y_min, y_max = x_test[:,1].min() - 1, x_test[:,1].max() + 1
    X, Y = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))
    F=LinReg.beta[2]*Y + LinReg.beta[1]*X + LinReg.beta[0] - 0.5
    ax2.contour(X,Y,F,[0],colors="black")
    ax2.contourf(X, Y, F, 0, alpha=0.5, colors=["darksalmon","lightblue"])
    # Drawing point for test set
    ax2.scatter(x_test[y_test.reshape(-1)==0,0],x_test[y_test.reshape(-1)==0,1],c="saddlebrown",label="y=0")
    ax2.scatter(x_test[y_test.reshape(-1)==1,0],x_test[y_test.reshape(-1)==1,1],c="navy",label="y=1")
    ax2.set_title('Linear Regression on test data '+dataset)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    y_pred_train=LinReg.predict(x).flatten()
    acc_train=(np.sum(y_pred_train==y.flatten())/y.shape[0])*100
    print("\t LinReg Train Accuracy :",np.round(acc_train,2),end="\t\t\t\t\t ")
    
    y_pred_test=LinReg.predict(x_test).flatten()
    acc_test=(np.sum(y_pred_test==y_test.flatten())/y_test.shape[0])*100
    print("LinReg Test Accuracy :",np.round(acc_test,2),"\n") 
    print("\t LinReg Train Misclassification error:",np.round(100-acc_train,2),end="\t\t")
    print("\t LinReg Test Misclassification error:",np.round(100-acc_test,2),end="\t\t")
    
    # Saving Figures
    if saving:
        import os
        if not os.path.isdir("images/"): 
            os.makedirs("images/")
        extent = ax1.get_window_extent().transformed(f.dpi_scale_trans.inverted())
        f.savefig('images/Linear_'+dataset+'.png', bbox_inches=extent.expanded(1.075, 1.2))
        extent = ax2.get_window_extent().transformed(f.dpi_scale_trans.inverted())
        f.savefig('images/Linear_test_'+dataset+'.png', bbox_inches=extent.expanded(1.1, 1.2))
        

# Class for Quadratic Discriminant Analysis :
class QuadraticDiscriminantAnalysis():

    def __init__(self,):
        '''
        Attributes:
        
        mu_0: np.array
            mean of p(x/y=0)
        mu_1: np.array
            mean of p(x/y=1)
        pi: float
            parameter of p(y)   
        sigma_1: np.array
            covariance matrix of p(x/y=1)  
        sigma_0: np.array
            covariance matrix of p(x/y=0)  
        '''      
        self.mu_0 = None
        self.mu_1 = None
        self.pi = None
        self.sigma_1 = None
        self.sigma_0 = None

    def fit(self, X, Y):
        """ Fit the data (X, Y).
    
        Parameters:
        -----------
        X: (num_samples, num_features) np.array
            Design matrix
        Y: (num_sampes, ) np.array
            Output vector
        Note:
        -----
        Compute parameters
        """
        self.pi = np.mean(Y)
        self.mu_0 = np.sum((1-Y)*X , axis=0)/np.sum(1-Y)
        self.mu_1 = np.sum(Y*X, axis=0)/np.sum(Y)
        self.sigma_0 = (np.dot((X-self.mu_0).T,(1-Y)*(X-self.mu_0)))/np.sum(1-Y)
        self.sigma_1 = (np.dot((X-self.mu_1).T,Y*(X-self.mu_1)))/np.sum(Y)

        
    def predict(self, X):
        """ Make predictions for data X.
    
        Parameters:
        -----------
        X: (num_samples, num_features) np.array
            New matrix
        
        Returns:
        -----
        y_pred: (num_samples, ) np.array
            Predictions
        """
        a = np.log(self.pi/(1-self.pi)) + 0.5*(np.dot(self.mu_0.T,np.dot(np.linalg.inv(self.sigma_0),self.mu_0))- np.dot(self.mu_1.T,np.dot(np.linalg.inv(self.sigma_1),self.mu_1)))    
        a = a +  0.5*np.log(np.linalg.det(self.sigma_0)/np.linalg.det(self.sigma_1))
        b = np.dot(self.mu_1.T,np.linalg.inv(self.sigma_1)) -  np.dot(self.mu_0.T,np.linalg.inv(self.sigma_0))
        c = 0.5*(np.linalg.inv(self.sigma_0) - np.linalg.inv(self.sigma_1))
        P =  a + np.dot(X,b.T) + np.sum(np.dot(X,c)*X,axis=1)
        P = 1/(1+np.exp(-P))
        return np.where(P<=0.5,0,1)
    
    
def plot_contour_QDA(QDA,dataset,x_train,y_train,x_test,y_test,saving=True) : 
    #if dataset == 'A' : 
    #    x,y,x_test,y_test = x_train_A,y_train_A,x_test_A,y_test_A
    #elif dataset == 'B' :
    #    x,y,x_test,y_test = x_train_B,y_train_B,x_test_B,y_test_B
    #else : 
    #    x,y,x_test,y_test = x_train_C,y_train_C,x_test_C,y_test_C

    x,y,x_test,y_test = x_train,y_train,x_test,y_test    
    
    pi , mu_0 , mu_1 , sigma_0 , sigma_1 = QDA.pi , QDA.mu_0, QDA.mu_1 , QDA.sigma_0 , QDA.sigma_1 
    
    a = np.log(pi/(1-pi)) + 0.5*(np.dot(mu_0.T,np.dot(np.linalg.inv(sigma_0),mu_0))- np.dot(mu_1.T,np.dot(np.linalg.inv(sigma_1),mu_1)))
    a += 0.5*np.log(np.linalg.det(sigma_0)/np.linalg.det(sigma_1))
    b = np.dot(mu_1.T,np.linalg.inv(sigma_1)) -  np.dot(mu_0.T,np.linalg.inv(sigma_0))
    c = 0.5*(np.linalg.inv(sigma_0) - np.linalg.inv(sigma_1))
    
    f, (ax1, ax2) = plt.subplots(1,2, figsize=(14,5))
        
    # Drawing the contour p(y=1/x)=0.5
    x_min, x_max = x[:,0].min() - 1,x[:,0].max() + 1
    y_min, y_max = x[:,1].min() - 1,x[:,1].max() + 1
    X, Y = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))
    F= c[1,1]*Y**2 + (b[1]+(c[1,0]+c[0,1])*X)*Y + c[0,0]*X**2+b[0]*X + a 
    ax1.contour(X,Y,F,[0],colors="black")
    ax1.contourf(X, Y, F, 0, alpha=0.5, colors=["darksalmon","lightblue"])
    # Drawing point for test set
    ax1.scatter(x[y.reshape(-1)==0,0],x[y.reshape(-1)==0,1],c="saddlebrown",label="y=0")
    ax1.scatter(x[y.reshape(-1)==1,0],x[y.reshape(-1)==1,1],c="navy",label="y=1")
    ax1.set_title('QDA on train data '+dataset)
    ax1.legend()
    

    # Drawing the contour p(y=1/x)=0.5
    x_min, x_max = x_test[:,0].min() - 1,x_test[:,0].max() + 1
    y_min, y_max = x_test[:,1].min() - 1, x_test[:,1].max() + 1
    X, Y = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))
    F= c[1,1]*Y**2 + (b[1]+(c[1,0]+c[0,1])*X)*Y + c[0,0]*X**2+b[0]*X + a 
    ax2.contour(X,Y,F,[0],colors="black")
    ax2.contourf(X, Y, F, 0, alpha=0.5, colors=["darksalmon","lightblue"])
    # Drawing point for test set
    ax2.scatter(x_test[y_test.reshape(-1)==0,0],x_test[y_test.reshape(-1)==0,1],c="saddlebrown",label="y=0")
    ax2.scatter(x_test[y_test.reshape(-1)==1,0],x_test[y_test.reshape(-1)==1,1],c="navy",label="y=1")
    ax2.set_title('QDA on test data '+dataset)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    y_pred_train=QDA.predict(x)
    acc_train=(np.sum(y_pred_train==y.flatten())/y.shape[0])*100
    print("\t QDA Train Accuracy :",np.round(acc_train,2),end="\t\t\t\t\t ")
    
    y_pred_test=QDA.predict(x_test)
    acc_test=(np.sum(y_pred_test==y_test.flatten())/y_test.shape[0])*100
    print("QDA Test Accuracy :",np.round(acc_test,2))
    
    print("\t QDA Train Misclassification error:",np.round(100-acc_train,2),end="\t\t")
    print("\t QDA Test Misclassification error:",np.round(100-acc_test,2),end="\t\t")
    
    # Saving Figures
    if saving:
        import os
        if not os.path.isdir("images/"): 
            os.makedirs("images/")
        extent = ax1.get_window_extent().transformed(f.dpi_scale_trans.inverted())
        f.savefig('images/QDA_'+dataset+'.png', bbox_inches=extent.expanded(1.075, 1.2))
        extent = ax2.get_window_extent().transformed(f.dpi_scale_trans.inverted())
        f.savefig('images/QDA_test'+dataset+'.png', bbox_inches=extent.expanded(1.1, 1.2))