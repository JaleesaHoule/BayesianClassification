Code for Bayesian Classification of randomly generated samples of multivariate Gaussian data, created for pattern recognition class assignment


## General background and theory

The core of any Bayes classifier is Bayes formula, which states:
```math
\begin{equation}
        P(\omega_j | \textbf{x} ) = \frac{p(\textbf{x}|\omega_j)P(w_j)}{p(\textbf{x})} \\   
        \text{,    } \\
        p(\textbf{x}) = \sum^c_{j=1}p(\textbf{x}|\omega_j)P(w_j)

\end{equation}
```

Here, $P(\omega_j | \textbf{x} )$ is the conditional probability that data belongs to a certain class ($\omega_j$) given that it has features $\textbf{x}$(also called the posterior probability). Similarly, $p(\textbf{x}|\omega_j)$ is the conditional probability that the data has those ($\textbf{x}$) features given the data belongs to $\omega_j$ (i.e. likelihood probability). $P(w_j)$ is the prior probability of how likely data is to belong to a class $\omega_j$, and p($\textbf{x}$) is the probability density of $\textbf{x}$ (i.e., evidence). 

In designing a Bayes classifier for multivariate data, this formula will become the basis of our discriminant functions, which are a set of equations $g_i(\textbf{x})$ for $i=1,...,n$ through which we are able to assign \textbf{x} features to a class $\omega_i$ by choosing $\omega_i$ if $g_i(\textbf{x})> g_j(\textbf{x})$ for all $j \neq i$. We will first let 
```math
\begin{equation}
    g_i(\textbf{x}) = p(\textbf{x}|\omega_i)P(w_i).
\end{equation}
```
Then, taking the natural log of this equation gives 
```math
\begin{equation}
    g_i(\textbf{x}) = ln(p(\textbf{x}|\omega_i)) + ln(P(w_i)).
\end{equation}
```
This manipulation makes $g_i(\textbf{x})$ a monotonically increasing function, which helps to simplify the classification process. For this project, we are asked to classify two categories of data. Thus, a dichotomizer can be used to classify the data, and can be defined as:
```math
\begin{equation}
    g(\textbf{x}) = g_1(\textbf{x}) - g_2(\textbf{x}).
\end{equation}
```
In this case, we will choose $\omega_1$ if $g(\textbf{x})>0$ and $\omega_2$ if $g(\textbf{x})<0$. The corresponding decision boundary for the two categories can be found by setting $g_1(\textbf{x}) = g_2(\textbf{x})$.

Now, if we assume $p(\textbf{x}|\omega_j) \sim N(\mu_i,\Sigma_i)$, we can substitute the probability density function (pdf) of a normal distribution into Equation \ref{log_discriminant}. For a multivariate normal distribution,
```math
\begin{equation}
    N(\mu,\Sigma) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}exp[-\frac{1}{2}(\textbf{x}-\mu)^T\Sigma^{-1}(\textbf{x}-\mu)] \text{, } \textbf{x} \in R^d .
\end{equation}
```
Substituting this expression into Equation \ref{log_discriminant} gives the discriminant function:
```math
\begin{equation}
    g_i(\textbf{x}) = -\frac{1}{2}(\textbf{x}-\mu_i)^T\Sigma_i^{-1}(\textbf{x}-\mu_i) - \frac{d}{2}ln(2\pi)-\frac{1}{2}ln(|\Sigma_i|)+ln(P(\omega_i)).
\end{equation}
```
This equation is the general discriminant function used for multivariate Gaussian data, and is what will be used to classify the data generated in this assignment. As discussed in class, there are 3 different scenarios in which this equation can be further discussed: case I ($\Sigma_i = \sigma^2I$ for each $w_i$), case II ($\Sigma_i = \Sigma$ for each $\omega_i$), and case III ($\Sigma_i =$ arbitrary for each $\omega_i$). 

## Classifier design based on case I parameters

In the most simple case, (i.e., case I), we assume the features $\textbf{x}$ for each $\omega_i$ are uncorrelated with the same variance. If this is true, the discriminant can be greatly simplified (by ignoring the constants $\frac{d}{2}ln(2\pi)$ and $\frac{1}{2}ln(\Sigma_i)$) as  
```math
\begin{equation}
    \begin{aligned}
        g_i(\textbf{x}) = \frac{-||\textbf{x}-\mu||^2}{2\sigma^2} + ln(P(\omega_i)).
    \end{aligned}
\end{equation}
```
By expanding and further simplifying this equation, we can see that the discriminant function for case I is linear, with 
```math
\begin{equation}
    g_i(\textbf{x}) =\textbf{w}_i^T\textbf{x}+w_{i0} \\
\end{equation}
```
where: 
```math          
\begin{equation}
 \textbf{w}_i=\frac{1}{\sigma^2}\mu_i, \text{w}_{i0}=\frac{-1}{2\sigma^2}\mu_i^T\mu_i+ln(P(\omega_i)). \\
\end{equation}
```
For the data in sample set A, $\Sigma_1 = \Sigma_2 = \sigma^2$. Therefore, the conditions for case I are satisfied and we can use this simplified discriminant form to determine the decision boundary and classify the data from sample A. 

By calculating the values of $w_i$ and $w_{i0}$ for each given [$\mu_i$, $\Sigma_i$] in sample set A, we can construct two discriminant functions ($g_1(x)$ and $g_2(x)$) for each dataset. Then, we can set those two functions equal to each other in order to determine the decision boundary and classify the samples.\\

## Classifier design based on case III parameters

In the case of sample set B, the covariance matrices are not equal. Since $\Sigma_1 \neq \Sigma_2$ and all features in $\omega_2$ do not have the same variance, this data can not be classified using the discriminant found in case I. Instead, the optimum classifier for this data will be case III, which yields a quadratic discriminant function. By rewriting the general equation above and ignoring the constant $\frac{d}{2}ln(2\pi)$, the discriminant for case III can be written as

```math
\begin{equation}
    g_i(\textbf{x})=\textbf{x}^T\textbf{W}_i\textbf{x} + \textbf{w}_i^T\textbf{x}+w_{i0}
\end{equation}
```
```math
where: 
\begin{equation}
\textbf{W}_i=-\frac{1}{2}\Sigma_i^{-1}, \textbf{w}_i=\Sigma_i^{-1}\mu_i, \text{  and  } \text{w}_{i0}=-\frac{1}{2}\mu_i^T\Sigma_i^{-1}\mu_i-\frac{1}{2}ln(|\Sigma_i|)+ln(P(w_i)).
\end{equation}
```
It can be noted that for case III, we no longer make the assumption that the features are uncorrelated (i.e., the covariance matrices do not need to be diagonal). This is also true for case II, though we will not discuss case II any further since neither the data from sample set A or B fall into this category. For dataset B, the priors, misclassification rates, and probability of error will be computed the same way as is described in Part 1 above. 


## Estimating prior probabilities

For both sample set A and B, we are given the instruction to produce n samples from $\omega_1$ and m samples from $\omega_2$. We can use these numbers to determine the priors as $P(w_1)= \frac{n}{n+m}$ and $P(\omega_2)= \frac{m}{n+m}$. However, if we did not have an idea of how many samples came from each class, we could instead choose to estimate priors by them equal to each other (which may or may not be a valid assumption). 


## Euclidean distance classifier theory and assumptions

In the case that $\Sigma_i=\sigma^2I$ for all $\omega_i$ and $P(\omega_i)=P(\omega_j)$ for $i\neq j$, the discriminant function found above can be further simplified to what is known as the Euclidean distance classifier. In this case, $g_i(\textbf{x})$ can be described as:

```math
\begin{equation}
\begin{aligned}
     g_i(\textbf{x}) &= - ||\textbf{x} -\mu_i||^2 \\
     &= - (\textbf{x}-\mu_i)^T(\textbf{x}-\mu_i). \\
\end{aligned}
\end{equation}
```
For this to be an optimum classifier (i.e., error is minimized), all priors must be equal, and the covariance matrices for all $\omega_i$ must satisfy $\Sigma_i=\sigma^2I$. As is true for cases I, II, and III, the assumption that all features are normally distributed must also hold true for the Euclidean distance classifier to be optimum.


## Determining misclassification rates

This assignment asks us to report the misclassification rate of each class and the total misclassification rate. For determining misclassification rates, we can compare the true class that each sample came from with the expected class for each sample (determined by assigning each sample $\omega_1$ if $g(\textbf{x})>0$, else $\omega_2$). The misclassification rate for each class can then be determined as $n_i/N_i$, where $n_i$ is the number of samples misclassified and $N_i$ is the total number of samples produced from class $\omega_i$. The total misclassification rate can be determined as $(n_1 + n_2)/(N_1+N_2)$, i.e., the total number of samples misclassified divided by the total number of samples. These results should be our true empirical error rates for the given samples. \\

## Assessing probability of error (Bhattacharyya bound)

The theoretical probability error is often determined using the Bhattacharyya error bound, which is defined as 
```math
\begin{equation}
    P(error) \leq P^{\beta}(\omega_1)P^{1-\beta}(\omega_2)e^{-k(\beta)}
\end{equation}
```
where
```math
\begin{equation}
    k(\beta) = \frac{\beta(1-\beta)}{2}(\mu_1-\mu_2)^T[(1-\beta)\Sigma_1 + \beta\Sigma_2]^{-1}(\mu_1-\mu_2) + \frac{1}{2}ln\left(\frac{|(1-\beta)\Sigma_1+\beta\Sigma_2|}{|\Sigma_1|^{1-\beta}|\Sigma_2|^\beta}\right).
\end{equation}
```
For the Bhattacharyya error bound, $\beta$ is set to 0.5, as opposed to the Chernoff error bound, which finds the $\beta$ which minimizes the equation above. The probability of error found using the Bhattacharyya bound will never be the most accurate error bound due to the looser approximation of $\beta$, therefore the true error rate is expected to be less than or equal to the probability of error estimated through the Bhattacharyya error bound. 





