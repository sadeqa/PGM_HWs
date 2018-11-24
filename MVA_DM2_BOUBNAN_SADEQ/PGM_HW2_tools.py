import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
from scipy.stats import chi2
from scipy.spatial.distance import cdist

################################################################################################

class my_KMeans():
    """
    Class for K-means clustering
    """
    def __init__(self,n_clusters,RandomState=42,max_iter=200):
        '''
        Parameters & Attributes:
        
        n_clusters: integer
            number of clusters to form
        RandomState: interger
            determines random number generation for centroid initialization
        centers: np.array
            array containing the centroids of the clusters
        init_centers_: np.array
            array containing the initial centroids
        labels_: (n,) np.array
            array containing labels of each point
        distortion_: np.array
            array containing the sum of squared distances of samples to their closest centroid.
        max_iter: float
            maximum iterations for the convergence
        '''
        self.k_ = n_clusters
        self.RandomState_ = RandomState
        self.max_iter_ = max_iter
        self.centers = None
        self.init_centers_ = None
        self.labels_ = None
        self.distortion_ = None
        
       
    def fit(self, X):
        """ Generate the centroids
        that better fit the data
        
        Parameters:
        -----------
        X: (n, p) np.array
            Data matrix
        
        Returns:
        -----
        None
        """
        # Initialization of centroids
        rng=np.random.RandomState(self.RandomState_)
        self.init_centers_ = X[rng.choice(range(X.shape[0]),self.k_,replace=False)]
        self.init_labels_ = np.argmin(cdist(X,self.init_centers_),axis=1)
        self.init_distortion_ = np.sum(np.linalg.norm(X-self.init_centers_[self.init_labels_],axis=1)**2)
               
        # Parameters
        convergence = False
        self.centers = self.init_centers_.copy()
        self.labels_ = self.init_labels_.copy()
        distortion=self.init_distortion_.copy()
        it=0
        
        # Convergence
        while (not(convergence) and it<self.max_iter_):
            it+=1
            for k in range(self.k_): self.centers[k]=np.mean(X[self.labels_==k],axis=0)
            self.labels_ = np.argmin(cdist(X,self.centers),axis=1)
            self.distortion_=np.sum(np.linalg.norm(X-self.centers[self.labels_],axis=1)**2)
            
            if np.abs(distortion-self.distortion_)<1e-3:
                convergence=True
            
            centers=self.centers
            distortion=self.distortion_ 
        self.it_=it
            
################################################################################################

def plot_KMeans_clusters(X,n_clusters,RandomStates,saving=False):
    """
    Plot initial centers and labels and final ones for data X
    with a random initilisation set by the RandomState number
    """
    for R_State in RandomStates:
        #[5,8,21,27,29,42]
        myK=my_KMeans(n_clusters=n_clusters, RandomState=R_State)
        myK.fit(X)
        
        colors=["red"]*n_clusters

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
        ax1.scatter(X.T[0],X.T[1], marker="2", c=myK.init_labels_)
        ax1.scatter(myK.init_centers_.T[0],myK.init_centers_.T[1], marker="P", c=colors, s=200, label="Initial centers")
        ax1.set_title("Training Data with initial clusters centers & labels")
        ax1.text(0.5,-0.11, "Initial Distortion : {}".format(np.round(myK.init_distortion_,2)), size=12, ha="center",transform=ax1.transAxes, fontsize=16)
        ax1.legend()

        ax2.scatter(X.T[0],X.T[1], marker="2", c=myK.labels_)
        ax2.scatter(myK.centers.T[0],myK.centers.T[1], marker="X", c=colors, s=200, label="Final centers")
        ax2.set_title("Training Data with final clusters centers & labels")
        ax2.text(0.5,-0.11, "Final Distortion : {}".format(np.round(myK.distortion_,2)), size=12, ha="center",transform=ax2.transAxes, fontsize=16)
        ax2.legend()
        
        # Saving Figures
        if saving:
            import os
            if not os.path.isdir("images/"): 
                os.makedirs("images/")
            plt.savefig('images/KMeans_'+str(R_State)+'_all.png')
            extent = ax1.get_window_extent().transformed(f.dpi_scale_trans.inverted())
            f.savefig('images/KMeans_'+str(R_State)+'_init.png', bbox_inches=extent.expanded(1.23, 1.3))
            extent = ax2.get_window_extent().transformed(f.dpi_scale_trans.inverted())
            f.savefig('images/KMeans_'+str(R_State)+'_final.png', bbox_inches=extent.expanded(1.23, 1.3))

        plt.tight_layout()
        plt.show()

################################################################################################
        
class my_iso_GMM():
    """ 
    Gaussian Mixture Model with covariances matrices 
    proportional to the identity (Isotropic)
    """
    def __init__(self, n_clusters, iter_max=100, tol=1e-5, RandomState=24):
        '''
        Parameters & Attributes:
        
        k_: integer
            number of clusters
        mu_: np.array
            array containing means
        Sigma_: np.array
            array containing covariance matrices
        cond_prob_: (n, K) np.array
            conditional probabilities for all data points "p(z/x)"
        labels_: (n, ) np.array
            labels for data points
        pi_: (K,) np.array
            array containing parameters for the multinomial latent variables
        iter_max: float
            maximum iterations allowed for the EM convergence
        tol: float
            tolerence for checking the EM convergence
        alpha_: (p,) np.array
            array containing the proportionality coefficients of the covariance matrices
        RandomState: int
            determines random number generation for centroid initialization
        '''
        self.k_ = n_clusters
        self.mu_ = None
        self.Sigma_ = None
        self.tau_ = None
        self.labels_ = None
        self.pi_ = None
        self.iter_max_ = iter_max
        self.tol_ = tol
        self.alpha_ = None
        self.RandomState = RandomState
    
    def compute_tau(self, X, mu, Sigma):
        '''Compute the conditional probability matrix p(z/x)
        shape: (n, K)
        '''
        n,p=X.shape
        pdf_k=(lambda k,x : multivariate_normal(mu[k],Sigma[k]).pdf(x))          
        pdf_k_s=np.array([pdf_k(k,X) for k in range(self.k_)]).T
        return pdf_k_s*self.pi_/((np.sum(pdf_k_s*self.pi_,axis=1))[:,None])

    def E_step(self, X, mu, Sigma, tau):
        '''Compute the expectation of the complete loglikelihood to check increment'''
        n,p=X.shape
        pdf_k=(lambda k,x : multivariate_normal(mu[k],Sigma[k]).pdf(x))   
        pdf_k_s=np.array([pdf_k(k,X) for k in range(self.k_)]).reshape(self.k_,n).T
        return np.sum(tau*np.log(pdf_k_s) + tau*np.log(self.pi_))
    
    def compute_complete_likelihood(self, X, labels):
        """ Compute the complete likelihood for a given labels 
        
        Parameters:
        -----------
         X: (n, p) np.array
            Data matrix
        labels: (n, ) np.array
            Data labels
        
        Returns:
        -----
        complete likelihood       
        """        
        return self.E_step(X,self.mu_,self.Sigma_,np.eye(self.k_)[labels]) 
    
    def fit(self, X):
        """ Find the parameters mu_ and Sigma_
        that better fit the data
        
        Parameters:
        -----------
        X: (n, p) np.array
            Data matrix
        
        Returns:
        -----
        None
        """
        n=X.shape[0]
        p=X.shape[1]
                    
        # Initialization with Kmeans
        k_init=my_KMeans(n_clusters=self.k_, RandomState=self.RandomState)
        k_init.fit(X)
        self.labels_=k_init.labels_        
        self.pi_=np.unique(self.labels_,return_counts=True)[1]/n
        self.mu_=k_init.centers
        self.Sigma_=[np.matmul((X[k_init.labels_==k]-self.mu_[k]).T,(X[k_init.labels_==k]-self.mu_[k])/(n*self.pi_[k])) for k in range(self.k_)]
        self.Sigma_=[np.max(np.linalg.eigvals(self.Sigma_[k]))*np.eye(p) for k in range(self.k_)]
        
        converged=False
        it=0
        
        #First E-Step
        self.tau_ = self.compute_tau(X,self.mu_,self.Sigma_)
        En = self.E_step(X,self.mu_,self.Sigma_,self.tau_)

        while ((not converged) and it<self.iter_max_):
            #M-Step
            self.pi_=np.mean(self.tau_,axis=0)
            self.mu_=np.matmul(self.tau_.T,X)/(np.sum(self.tau_,axis=0).reshape(-1,1))
            self.alpha_= [np.dot(self.tau_[:,k],np.sum((X-self.mu_[k])*(X-self.mu_[k]),axis=1))/(p*np.sum(self.tau_[:,k])) for k in range(self.k_)]
            self.Sigma_=[alpha*np.eye(p) for alpha in self.alpha_]
            
            #E-Step
            Enp1=self.E_step(X,self.mu_,self.Sigma_,self.tau_)
            if (np.abs(Enp1/En-1))<self.tol_:
                converged=True
            it+=1
            En=Enp1
            self.tau_ = self.compute_tau(X,self.mu_,self.Sigma_)
        
        # Assigning labels
        self.labels_=np.argmax(self.tau_,axis=1) 
        
        # Computing complete likelihood
        self.complete_likelihood_ = self.compute_complete_likelihood(X,self.labels_)
        
    def predict(self, X):
        """ Predict labels for X
        
        Parameters:
        -----------
        X: (n, p) np.array
            New data matrix
        
        Returns:
        -----
        label assigment        
        """
        return np.argmax(self.compute_tau(X, self.mu_, self.Sigma_),axis=1)

    def predict_proba(self, X):
        """ Predict probability vector for X
        
        Parameters:
        -----------
        X: (n, p) np.array
            New data matrix
        
        Returns:
        -----
        proba: (n, k) np.array        
        """
        return compute_tau(X, self.mu_, self.Sigma_)

    
################################################################################################
    
class my_GMM():
    """ 
    Gaussian Mixture Model with general covariances matrices 
    """   
    def __init__(self, n_clusters, iter_max=100, tol=1e-5, RandomState=24):
        '''
        Parameters & Attributes:
        
        k_: integer
            number of clusters
        mu_: np.array
            array containing means
        Sigma_: np.array
            array containing covariance matrices
        tau_: (n, K) np.array
            conditional probabilities for all data points "p(z/x)"
        labels_: (n, ) np.array
            labels for data points
        pi_: (K,) np.array
            array containing parameters for the multinomial latent variables
        iter_max: float
            maximum iterations allowed for the EM convergence
        tol: float
            tolerence for checking the EM convergence
        RandomState: int
            determines random number generation for centroid initialization
        '''
        self.k_ = n_clusters
        self.mu_ = None
        self.Sigma_ = None
        self.tau_ = None
        self.labels_ = None
        self.pi_ = None
        self.iter_max_ = iter_max
        self.tol_ = tol
        self.RandomState=RandomState

    def compute_tau(self, X, mu, Sigma):
        '''Compute the conditional probability matrix p(z/x)
        shape: (n, K)
        '''
        n,p=X.shape
        pdf_k=(lambda k,x : multivariate_normal(mu[k],Sigma[k]).pdf(x))          
        pdf_k_s=np.array([pdf_k(k,X) for k in range(self.k_)]).T
        return pdf_k_s*self.pi_/((np.sum(pdf_k_s*self.pi_,axis=1))[:,None])

    def E_step(self, X, mu, Sigma, cond_prob):
        '''Compute the expectation of the complete loglikelihood to check increment'''
        n,p=X.shape
        pdf_k=(lambda k,x : multivariate_normal(mu[k],Sigma[k]).pdf(x))   
        pdf_k_s=np.array([pdf_k(k,X) for k in range(self.k_)]).reshape(self.k_,n).T
        return np.sum(cond_prob*np.log(pdf_k_s) + cond_prob*np.log(self.pi_))
    
    def compute_complete_likelihood(self, X, labels):
        """ Compute the complete likelihood for a given labels 
        
        Parameters:
        -----------
         X: (n, p) np.array
            Data matrix
        labels: (n, ) np.array
            Data labels
        
        Returns:
        -----
        complete likelihood       
        """        
        return self.E_step(X,self.mu_,self.Sigma_,np.eye(self.k_)[labels])    
        
    def fit(self, X):
        """ Find the parameters mu_ and Sigma_
        that better fit the data
        
        Parameters:
        -----------
        X: (n, p) np.array
            Data matrix
        
        Returns:
        -----
        None
        """
        n=X.shape[0]
        p=X.shape[1]        
            
        # Initialization with kmeans   
        k_init=my_KMeans(n_clusters=self.k_,RandomState=self.RandomState)
        k_init.fit(X)
        self.labels_=k_init.labels_
        self.pi_=np.unique(self.labels_,return_counts=True)[1]/n
        self.mu_=k_init.centers
        self.Sigma_=[np.matmul((X[k_init.labels_==k]-self.mu_[k]).T,(X[k_init.labels_==k]-self.mu_[k])/(n*self.pi_[k])) for k in range(self.k_)]            
        
        converged=False
        it=0
        
        #First E-Step
        self.tau_ = self.compute_tau(X,self.mu_,self.Sigma_)
        En=self.E_step(X,self.mu_,self.Sigma_,self.tau_)
        
        
        while ((not converged) and it<self.iter_max_):
            #M-Step
            self.pi_=np.mean(self.tau_,axis=0)
            self.mu_=np.matmul(self.tau_.T,X)/(np.sum(self.tau_,axis=0).reshape(-1,1))
            self.Sigma_=np.array([np.matmul((X-self.mu_[k]).T,((X-self.mu_[k])*self.tau_[:,k].reshape(-1,1)))/(np.sum(self.tau_[:,k])) for k in range(self.k_)])
            
            #E-Step
            Enp1=self.E_step(X,self.mu_,self.Sigma_,self.tau_)
            if (np.abs(Enp1/En-1)) < self.tol_:
                converged=True
            it+=1
            En=Enp1
            self.cond_prob_ = self.compute_tau(X,self.mu_,self.Sigma_)
        
        # Assigning labels
        self.labels_=np.argmax(self.tau_,axis=1)  
        
        # Computing complete likelihood
        self.complete_likelihood_ = self.compute_complete_likelihood(X,self.labels_)
        
    def predict(self, X):
        """ Predict labels for X
        
        Parameters:
        -----------
        X: (n, p) np.array
            New data matrix
        
        Returns:
        -----
        label assigment        
        """
        return np.argmax(self.compute_tau(X,self.mu_,self.Sigma_),axis=1)
    

    def predict_proba(self, X):
        """ Predict probability vector for X
        
        Parameters:
        -----------
        X: (n, p) np.array
            New data matrix
        
        Returns:
        -----
        proba: (n, k) np.array        
        """
        return self.compute_tau(X,self.mu_,self.Sigma_)

    
################################################################################################    
    
def plot_cov_elipse(ax,cov,mu,confidence=0.95,color="black"):
    """ Plot the ellipse that contains a confidence
    percentage of the mass of the Gaussian distribution N(mu,cov)
    """
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:,order]
    
    confid_coef=chi2(df=2).ppf(confidence)
    theta = np.degrees(np.arctan2(*vecs.T[0][::-1]))    
    w, h = 2 * np.sqrt(confid_coef*vals)
    
    ell = Ellipse(xy=(mu[0], mu[1]),
                  width=w, height=h,
                  angle=theta, color=color)
    ell.set_facecolor('none')
    ax.add_artist(ell)
    
################################################################################################ 

def plot_results(X,iso_model,iso_labels,general_model,general_labels,data_label,color_plot="orangered",confidence=0.9,saving=False):
    """ Plot the results of the Isotropic and General mixture models
    """
    colors=[color_plot]
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
    ax1.scatter(X.T[0],X.T[1], marker="2", c=iso_labels)
    ax1.scatter(iso_model.mu_.T[0],iso_model.mu_.T[1], c=colors, s=200, label="Centers")
    ax1.set_xlim(np.min(X)-0.5,np.max(X)+0.5)
    ax1.set_ylim(np.min(X)-0.5,np.max(X)+0.5)
    for sigma,mu in list(zip(iso_model.Sigma_,iso_model.mu_,)):
        plot_cov_elipse(ax1,sigma,mu,color="black",confidence=confidence)
    ax1.text(0.5,-0.11, "Loglikelihood : {}".format(np.round(iso_model.compute_complete_likelihood(X,iso_labels),2)), size=12, ha="center",transform=ax1.transAxes, fontsize=16)
    ax1.set_title("{} - Isotropic GMM - {}% confidence ellipses".format(data_label,int(confidence*100)), fontsize=13)
    ax1.legend()

    ax2.scatter(X.T[0],X.T[1], marker="2", c=general_labels)
    ax2.scatter(general_model.mu_.T[0],general_model.mu_.T[1], c=colors, s=200, label="Centers")
    ax2.set_xlim(np.min(X)-0.5,np.max(X)+0.5)
    ax2.set_ylim(np.min(X)-0.5,np.max(X)+0.5)
    for sigma,mu in list(zip(general_model.Sigma_,general_model.mu_)):
        plot_cov_elipse(ax2,sigma,mu,color="black",confidence=confidence)
    ax2.text(0.5,-0.11, "Loglikelihood : {}".format(np.round(general_model.compute_complete_likelihood(X,general_labels),2)), size=12, ha="center",transform=ax2.transAxes, fontsize=16)
    ax2.set_title("{} - General GMM - {}% confidence ellipses".format(data_label,int(confidence*100)), fontsize=13)
    ax2.legend()
    
    # Saving Figures
    if saving:
        import os
        if not os.path.isdir("images/"): 
            os.makedirs("images/")   
        data_label='_'.join(data_label.split())
        extent = ax1.get_window_extent().transformed(f.dpi_scale_trans.inverted())
        f.savefig('images/'+data_label+'_ISO_'+'.png', bbox_inches=extent.expanded(1.25, 1.24))
        extent = ax2.get_window_extent().transformed(f.dpi_scale_trans.inverted())
        f.savefig('images/'+data_label+'_GENERAL_'+'.png', bbox_inches=extent.expanded(1.25, 1.24))

    plt.tight_layout()
    plt.show()