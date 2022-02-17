import numpy as np

def my_imfilter(image, filter):
  """
  Apply a filter to an image. Return the filtered image.

  Args
  - image: numpy nd-array of dim (m, n, c)
  - filter: numpy nd-array of dim (k, k)
  Returns
  - filtered_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You may not use any libraries that do the work for you. Using numpy to work
   with matrices is fine and encouraged. Using opencv or similar to do the
   filtering for you is not allowed.
  - I encourage you to try implementing this naively first, just be aware that
   it may take an absurdly long time to run. You will need to get a function
   that takes a reasonable amount of time to run so that I can finish grading 
   before the heat death of hte universe. 
   your code works.
  - Remember these are RGB images, accounting for the final image dimension.
  """

  assert filter.shape[0] % 2 == 1
  assert filter.shape[1] % 2 == 1

  ############################
  ### TODO: YOUR CODE HERE ###

  k_height = np.size(filter,0)
  k_width = np.size(filter,1)
  height = np.size(image,0)
  width = np.size(image,1)
  ceng = np.size(image,2)
  #print(height,width,ceng)
  filtered = np.zeros([height,width,ceng])
  big_form = another_form(image,filter)
  i = 0
  c = 0
  while c<ceng:
      i = 0
      while i < height:
          j = 0
          while j < width:
              box = 0
              a = 0
              while a < k_height:
                  b = 0
                  while b < k_width:
                      box+=big_form[i+a,j+b,c]*filter[a,b]
                      b+=1
                  a+=1
              filtered[i,j,c] = box
              j+=1
          i+=1
      c+=1
    
            
  return filtered
  raise NotImplementedError('`my_imfilter` function in `project2.py` ' +
    'needs to be implemented')


  ### END OF STUDENT CODE ####
  ############################


def another_form(image,kernel):
    """
    a helper function for my_imfilter() function
    to create a bigger map so that matching with the kernel scale
    """
    k_height = np.size(kernel,0)
    k_width = np.size(kernel,1)
    #print(k_height,k_width)
    height = np.size(image,0)
    width = np.size(image,1)
    ceng = np.size(image,2)
    transformed = np.zeros([height+(k_height-1)*2,width+(k_width-1)*2,ceng])
    #print(np.size(filtered,0),np.size(filtered,1))
    c = 0
    while c< ceng:
        m = 0
        while m < height:
            n = 0
            while n < width:
                transformed[m+(k_height-1),n+(k_width-1),c] = image[m,n,c]
                n+=1
            m+=1
        c+=1
    #print(transformed)
    return transformed


def create_hybrid_image(image1, image2, filter):
  """
  Takes two images and creates a hybrid image. Returns the low
  frequency content of image1, the high frequency content of
  image 2, and the hybrid image.

  Args
  - image1: numpy nd-array of dim (m, n, c)
  - image2: numpy nd-array of dim (m, n, c)
  Returns
  - low_frequencies: numpy nd-array of dim (m, n, c)
  - high_frequencies: numpy nd-array of dim (m, n, c)
  - hybrid_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You will use your my_imfilter function in this function.
  - You can get just the high frequency content of an image by removing its low
    frequency content. Think about how to do this in mathematical terms.
  - Don't forget to make sure the pixel values are >= 0 and <= 1. This is known
    as 'clipping'.
  - If you want to use images with different dimensions, you should resize them
    in the notebook code.
  """

  assert image1.shape[0] == image2.shape[0]
  assert image1.shape[1] == image2.shape[1]
  assert image1.shape[2] == image2.shape[2]

  ############################
  ### TODO: YOUR CODE HERE ###
  low_frequencies = my_imfilter(image1,filter)
  high_low = my_imfilter(image2,filter)
  k_height = np.size(filter,0)
  k_width = np.size(filter,1)
  height = np.size(image2,0)
  width = np.size(image2,1)
  ceng = np.size(image2,2)
  #print(height,width,ceng)
  high_frequencies = np.zeros([height,width,ceng])
  big_form = another_form(image2,filter)
  i = 0
  c = 0
  while c<ceng:
      i = 0
      while i < height:
          j = 0
          while j < width:
              box = 0
              high_frequencies[i,j,c] = image2[i,j,c]-high_low[i,j,c]
              j+=1
          i+=1
      c+=1
  
  hybrid = np.zeros([height,width,ceng])
  d = 0
  while d < ceng:
        e = 0
        while e < height:
            f = 0
            while f < width:
                low = low_frequencies[e,f,d]
                high = high_frequencies[e,f,d]
                hybrid[e,f,d] = low + high
                f+= 1
            e+=1
        d += 1
  return (low_frequencies,high_frequencies,hybrid)
  

  raise NotImplementedError('`create_hybrid_image` function in ' + 
    '`student_code.py` needs to be implemented')

  ### END OF STUDENT CODE ####
  ############################

  return low_frequencies, high_frequencies, hybrid_image
