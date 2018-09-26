import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  C = 0.01 # a constant term to prevent numerical instability  
  for i in range(X.shape[0]):
    numerators = np.exp(X[i,:].dot(W) + np.log(C) )
    denom = np.sum(numerators)
    y_onehot = (np.arange(W.shape[1])==y[i]).astype(np.int32)
    #print(numerators.shape,y_onehot.shape)
    y_pred = numerators/denom
    #print(y_pred)
    #print(y_pred - y_onehot)
    dW += np.outer( X[i,:], (y_pred - y_onehot))
    
    loss -= np.sum(y_onehot*np.log(y_pred))
    
  loss /= X.shape[0]
  loss += reg*np.sum(W**2)/2
    
  dW /= X.shape[0]
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  C = 0.01 # a constant term to prevent numerical instability

  numerators = np.exp( X.dot(W) + np.log(C) )
  denoms = np.sum(numerators, axis = 1 )
  y_pred = numerators/denoms[:,None]
  y_onehot = np.equal(y[:,None], np.arange(W.shape[1])[:,None].T )
  
  #print(y_onehot.shape,y_pred.shape)
  loss = -np.sum(y_onehot*np.log(y_pred))/X.shape[0] + reg*np.sum(W**2)/2

  dW = X.T.dot(y_pred-y_onehot)/X.shape[0] + reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

