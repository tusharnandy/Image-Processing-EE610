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
