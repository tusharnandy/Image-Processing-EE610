from skimage import color
import numpy as np

def hist_equalize(image_rgb):
  HSV = color.rgb2hsv(image_rgb) # converting RGB to HSV format
  H,S,V = HSV[:,:,0], HSV[:,:,1], HSV[:,:,2] # separating the H,s and V layers
  # The 'V' layer is scaled from [0 - 255] to [0 - 1], we need to get the appropriate bins back
  V = np.round(V*255) # Scaling the intensity values

  bins = [i for i in range(256)] # an array of intensity values
  counts = [] # to store the counts
  for i in bins:
    counts.append((V==i).sum()) # collecting the count for each instensity level

  probs = counts/sum(counts)  # converting to probabilites
  cdf = np.array([probs[:i].sum() for i in range(256)]) # generating the cdf by partial summation

  # We now have a map for each pixel
  # The index value is the key and the corresponding array element
  # is the new instensity level for that pixel
  T_r = np.round(cdf*255)

  V_new = np.array([T_r[int(i)] for i in V.flatten()]) # Transforming each pixel to new intensity value
  V_new = V_new.reshape(V.shape) # reshaping the matrix

  HSV_new = np.zeros(HSV.shape) # Forming a new HSV tensor
  HSV_new[:,:,0], HSV_new[:,:,1], HSV_new[:,:,2] = H, S, V_new/255 # Appropriately scaled values before returning
  new_image = color.hsv2rgb(HSV_new)*255

  return new_image.astype("uint8")



def log_transform(img):
  HSV = color.rgb2hsv(img) # converting RGB to HSV format
  H,S,V = HSV[:,:,0], HSV[:,:,1], HSV[:,:,2] # separating the H,s and V layers
  # The 'V' layer is scaled from [0 - 255] to [0 - 1], we need to get the appropriate bins back
  V = np.round(V*255) # Scaling the intensity values

  c = 255/np.log(256) # computing the scaling factor such that 255 maps to 255 only
  T_r = np.array([np.round(c*np.log(1+r)) for r in range(256)]) # This array maps original pixel value to gamma corrected value

  V_new = np.array([T_r[int(i)] for i in V.flatten()]).reshape(V.shape) # Creating the new 'V' matrix

  HSV_new = np.zeros(HSV.shape) # Forming a new HSV tensor
  HSV_new[:,:,0], HSV_new[:,:,1], HSV_new[:,:,2] = H, S, V_new/255 # Appropriately scaled values before returning
  new_image = color.hsv2rgb(HSV_new)*255

  return new_image.astype("uint8")



def gamma_corrected(img, gamma):
  HSV = color.rgb2hsv(img) # converting RGB to HSV format
  H,S,V = HSV[:,:,0], HSV[:,:,1], HSV[:,:,2] # separating the H,s and V layers
  # The 'V' layer is scaled from [0 - 255] to [0 - 1], we need to get the appropriate bins back
  V = np.round(V*255) # Scaling the intensity values

  k = 255.0/(255.0**gamma) # computing the scaling factor such that 255 maps to 255 only
  T_r = np.array([np.round(k*(i**gamma)) for i in range(256)]) # This array maps original pixel value to gamma corrected value

  V_new = np.array([T_r[int(i)] for i in V.flatten()]).reshape(V.shape) # Creating the new 'V' matrix

  HSV_new = np.zeros(HSV.shape) # Forming a new HSV tensor
  HSV_new[:,:,0], HSV_new[:,:,1], HSV_new[:,:,2] = H, S, V_new/255 # Appropriately scaled values before returning
  new_image = color.hsv2rgb(HSV_new)*255

  return new_image.astype("uint8")


def gaussian_2d(x,y,sigma):
  numerator = np.exp(-(x*x + y*y)/(2*sigma*sigma))
  denominator = 2*np.pi*sigma*sigma
  return numerator/denominator

def gaussian_filter(n, sigma=3):
  if n%2==0:
    n = n+1
  center = (n-1)/2
  filter = np.array([[gaussian_2d(j-center,i-center,sigma) for i in range(n)] for j in range(n)])
  return filter/filter.sum()

def zero_padding(matrix, kernel):
  k = int((kernel-1)/2)
  m, n = matrix.shape[:2]
  padded_matrix = np.zeros((m+k*2, n+k*2))
  padded_matrix[k:m+k, k:n+k] = matrix.copy()
  return padded_matrix

def img_filter(padded_img, filter):
  k = filter.shape[0]
  k = int((k-1)/2)
  m, n = padded_img.shape
  new_img = np.zeros(padded_img.shape)

  for i in range(k,m-k):
    for j in range(k,n-k):
      sub_img = padded_img[i-k:i+k+1, j-k:j+k+1].copy()
      filtered_sub_img = sub_img.flatten() * filter.flatten()
      filtered_sub_img = filtered_sub_img.reshape((2*k+1,-1))
      new_img[i,j] = filtered_sub_img.sum()

  return new_img[k:m-k, k:n-k]

def gaussian_blur(img, kernel_size=51, sigma=5):
    HSV = color.rgb2hsv(img) # converting RGB to HSV format
    H,S,V = HSV[:,:,0], HSV[:,:,1], HSV[:,:,2] # separating the H,s and V layers
    V_new = img_filter(zero_padding(V, kernel_size), gaussian_filter(kernel_size,sigma)).copy() # applying blurring to V
    HSV_new = np.zeros(HSV.shape) # initializing a new HSV tensor
    HSV_new[:,:,0], HSV_new[:,:,1], HSV_new[:,:,2] = H, S, V_new # assigning the appropriate values
    img_new = color.hsv2rgb(HSV_new)    # coverting hsv back to rgb
    img_new = np.round(img_new*255)     # for display, the range of values should be in [0..255]
    img_new = img_new.astype('uint8')   # and the format should be "uint8" for byte-sized pixel

    return img_new

def laplacian_filter(n=3, diagonal=False):
  center = int((n-1)/2)
  filter = np.ones((n,n))

  if not diagonal:
    filter[0,0] = 0
    filter[n-1, 0] = 0
    filter[0, n-1] = 0
    filter[n-1, n-1] = 0

  filter[center, center] -= filter.sum()

  return -1*filter/3

def sharpen_lap(img, correction=1):
  HSV = color.rgb2hsv(img)
  H, S, V = HSV[:,:,0], HSV[:,:,1], HSV[:,:,2]
  # V = np.round(V*255) # Scaling the intensity values

  kernel_size = 3
  V_new = img_filter(zero_padding(V, kernel_size), laplacian_filter(kernel_size)).copy()

  # if correction > 1:
  #   correction = 1
  V_sharp = V + correction*V_new

  HSV_new = np.zeros(HSV.shape)
  HSV_new[:,:,0], HSV_new[:,:,1], HSV_new[:,:,2] = H, S, V_sharp
  img_new = color.hsv2rgb(HSV_new)

  img_new = np.where(img_new < 0, 0, img_new)
  img_new = np.where(img_new > 1, 1, img_new)
  img_new = (img_new*255).astype("uint8")

  return img_new

def brightness(img, beta):
  img_new = img.astype("float64") + beta
  img_new = trim(img_new, min=0, max=255)
  img_new = np.round(img_new).astype("uint8")
  return img_new

def contrast(img, alpha):
  img_new = img.astype("float64")*alpha
  img_new = trim(img_new, min=0, max=255)
  img_new = np.round(img_new).astype("uint8")
  return img_new

def trim(var, min=0, max=1):
  var = np.where(var<min, min, var)
  var = np.where(var>max, max, var)
  return var
