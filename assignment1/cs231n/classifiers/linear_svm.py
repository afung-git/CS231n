import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i,:].T
        dW[:,y[i]] -= X[i,:].T

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)/2
  dW += reg*W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
 
  score = X.dot(W)
  # Get the score of the correct class, for each example
  # Advanced indexing: score[ 0:(num_examples-1) , y ] ;score NxC
  test = score[np.arange(y.shape[0]), y]
  
  # Create a matrix for the correct score. This will be added to each and every
  # class.
  # First create a list of vectors  
  # Use vstack as hstack will append them.
  # Transpose to get back the right dimensions 
  correct = np.vstack([test]*W.shape[1]).T
  
  # Assign correct scores with very large negative values. so the max function
  # will always output a zero for them (We don't add the correct score to loss) 
  score[np.arange(y.shape[0]),y] = -1e6
  
  # Compute hinge loss= sj -si + delta (j =/= i)
  hinge = score - correct + 1
  # Mask to make negatives 0
  mask = (hinge<0)
  # Mask for gradient computation
  grad_mask = (hinge>0).astype(np.int32)
  # Set negatives to 0
  hinge[mask] = 0
  # Sum up all hinge losses, divide by num_examples + lambda*sum(W**2)
  loss = np.sum(hinge)/y.shape[0] + reg*np.sum(W*W)/2
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # for each incorrect class that has its score too close to the correct class's score,
  # there will be hinge loss. So there will be sj-si. Diff wrt si and we have -x
  # so dw wrt the correct score (si) is the sum of -x for each violating incorrect class
  # dw wrt to the violating incorrect class is just -x (Diff wrt sj = x), just a single -x

  # How much X is added the correct class, for each eg, vector form
  addtocorrect = np.sum(grad_mask,axis=1)
  
  # Mistake made 22/09/2018. Debug with a copy of the gradient mask  
  grad_mask_2 = np.copy(grad_mask)
  # If you index addtocorrect with y, y ranges from 0-9 (num_classes), 
  # so it only takes addtocorrect[0-9] (first 10 examples' total # of violations) 
  # which happens to always be -9
  grad_mask_2[np.arange(y.shape[0]),y] = -addtocorrect[y]
  # Should index with -addtocorrect[0:num_examples-1], since each example has its
  # own total number of violations to add
  grad_mask[np.arange(y.shape[0]),y] = -addtocorrect[np.arange(y.shape[0])]
  
  # Now, gradmask has either 0 or 1 for incorrect scores. For correct scores,
  # it will have between -9 to 0 (0 is very unlikely, as it means all incorrect
  # classes have lower scores than the correct class) 

  # Evidence
  #print(y)
  #print(-addtocorrect[np.arange(y.shape[0])])
  #print(-addtocorrect[y])
  #print(np.linalg.norm(grad_mask-grad_mask_2,ord='fro')) 
  #print(grad_mask[:20])
  #print(grad_mask_2[:20])  
  #print(grad_mask.shape, grad_mask_2.shape)
  
  # dW(i,j) = how much to tune i-th feature, j-th class's weight [DxC]
  # grad_mask(i,j) = how many Xs for i-th example, j-th class [NxC] (can be 
  # used to scale any n-th feature in the i-th example)
    
  # Why can scale any and all features for one example? In the loop approach,
  # add one whole X feature-column to a feature-column of dW. 
  # dW[:,j] += X[i,:].T where X[i,:] is the i-th example feature vector
  # dW[:,j] means all features in j-th class
  # same concept, just that the scale is 1 or 0 in loop approach. grad_mask=0,1
    
  # eg. dW(1,2) = X.T(1,:)*grad_mask(:,2) (2nd feature, all examples)*(all examples, 3rd class)
  # dW(1,2) = ...+ X.T(1, i-th) * grad_mask(i-th, 2) +... as seen, the same grad_mask element scales the 2nd &
  # dW(2,2) = ...+ X.T(2, i-th) * grad_mask(i-th, 2) +... 3rd features of the i-th example
  dW = np.matmul(X.T,grad_mask)

  dW /= y.shape[0]
    
  dW += reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
