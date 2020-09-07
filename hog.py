#!/usr/bin/env python
# coding: utf-8

# In[23]:


from skimage import feature
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[24]:


get_ipython().system('which python')


# In[47]:


get_ipython().system('pip install opencv-rolling-ball')


# In[31]:


image_raw = cv2.imread('02_80-100s__crop256px_5_5.bmp')
image_gb3 = cv2.GaussianBlur(image_raw,(3,3), cv2.BORDER_DEFAULT)
image_gb5 = cv2.GaussianBlur(image_raw,(5,5), cv2.BORDER_DEFAULT)


# In[33]:


fig, (ax0, ax1, ax2) = plt.subplots(nrows=1,
                                    ncols=3,
                                    figsize=(8, 2.5),
                                    sharex=True,
                                    sharey=True)
ax0.imshow(image_raw, cmap='gray')
ax0.set_title('original image')
ax0.axis('off')

ax1.imshow(image_gb3, vmin=image.min(), vmax=image.max(), cmap='gray')
ax1.set_title('gaussian blur 3')
ax1.axis('off')

ax2.imshow(image_gb5, vmin=image.min(), vmax=image.max(), cmap='gray')
ax2.set_title('gaussian blur 5')
ax2.axis('off')


fig.tight_layout()
fig.savefig('image-gaussian_blur.png')


# In[35]:


#hog feature
(hog, hog_image) = feature.hog(image_gb3, orientations=9, 
                    pixels_per_cell=(8, 8), cells_per_block=(4, 4), 
                    block_norm='L2-Hys', visualize=True, transform_sqrt=True)

plt.imshow(image_raw)
plt.imshow(hog_image,alpha=0.7, cmap='hsv')


plt.show()
#plt.imwrite('hog_gb.jpg', hog_image*255)


# In[162]:


img_rgb = image_gb5


# In[163]:


plt.imshow(img_rgb)
plt.show()


# In[164]:


from cv2_rolling_ball import subtract_background_rolling_ball
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)


# In[183]:


plt.imshow(img_gray)
plt.show()


# In[180]:


img_subt = img_gray
img_subt, img_bkg = subtract_background_rolling_ball(img_subt,4, light_background=False, 
                                                    use_paraboloid=True, do_presmooth=True)


# In[216]:


img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
plt.imshow(img_gray)
plt.show()
plt.imshow(img_bkg)
plt.show()
plt.imshow(img_subt)
plt.show()


# In[217]:


fig, (ax0, ax1, ax2) = plt.subplots(nrows=1,
                                    ncols=3,
                                    figsize=(8, 2.5),
                                    sharex=True,
                                    sharey=True)
ax0.imshow(img_gray)
ax0.set_title('original image')
ax0.axis('off')

ax1.imshow(img_bkg)
ax1.set_title('img_subt')
ax1.axis('off')

ax2.imshow(img_subt)
ax2.set_title('img_bkg')
ax2.axis('off')


fig.tight_layout()
fig.savefig('image-ball.png')


# In[188]:


from scipy.ndimage import gaussian_filter
from skimage import data
from skimage import img_as_float
from skimage.morphology import reconstruction


# In[189]:


# Convert to float: Important for subtraction later which won't work with uint8
image_f = img_as_float(image_raw)
image = gaussian_filter(image_f, 2.5)

seed = np.copy(image)
seed[1:-1, 1:-1] = image.min()
mask = image

dilated = reconstruction(seed, mask, method='dilation')


# In[193]:


# filtering regional maxima
fig, (ax0, ax1, ax2) = plt.subplots(nrows=1,
                                    ncols=3,
                                    figsize=(8, 2.5),
                                    sharex=True,
                                    sharey=True)

ax0.imshow(image, cmap='gray')
ax0.set_title('original image')
ax0.axis('off')

ax1.imshow(dilated, vmin=image.min(), vmax=image.max(), cmap='gray')
#ax1.imshow(dilated,vmin=0,vmax=255,cmap='gray')
ax1.set_title('dilated')
ax1.axis('off')
subt=image-dilated
ax2.imshow(subt, cmap='jet')
ax2.set_title('image - dilated')
ax2.axis('off')

fig.tight_layout()
fig.savefig('image-dilated.png')


# In[212]:


from skimage.color import rgb2gray
plt.imshow(subt)
plt.show()
subt_gray = rgb2gray(subt)
plt.imshow(subt_gray, vmin=subt_gray.min(), vmax=subt_gray.max())
plt.show()


# In[213]:


subt.max()


# In[40]:


subt_ubyte = img_as_ubyte(subt)


# In[41]:


subt_ubyte;


# In[42]:


subt_ubyte.max()


# In[43]:


type(subt_ubyte)


# In[198]:


plt.imshow(subt_ubyte)
plt.show()


# In[45]:


h = image.min()
seed = image - h
dilated = reconstruction(seed, mask, method='dilation')
hdome = image - dilated


# In[ ]:





# In[46]:


fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(8, 2.5))
yslice = 120

ax0.plot(mask[yslice], '0.5', label='mask')
ax0.plot(seed[yslice], 'k', label='seed')
ax0.plot(dilated[yslice], 'r', label='dilated')
ax0.set_ylim(-0.2, 2)
ax0.set_title('image slice')
ax0.set_xticks([])
ax0.legend()

ax1.imshow(dilated, vmin=image.min(), vmax=image.max(), cmap='gray')
ax1.axhline(yslice, color='r', alpha=0.4)
ax1.set_title('dilated')
ax1.axis('off')

ax2.imshow(hdome, cmap='gray')
ax2.axhline(yslice, color='r', alpha=0.4)
ax2.set_title('image - dilated')
ax2.axis('off')

fig.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




