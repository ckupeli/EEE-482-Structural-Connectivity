import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import nibabel as nib
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.segment.mask import median_otsu

filename = 'DTI30s010'

# Get directory of DWI
fimg = join('./', filename + '.nii') # 4D NIfTI1 file
print(fimg)
fbval = join('./', filename + '.bval') # b-values
print(fbval)
fbvec = join('./', filename + '.bvec') # b-vectors
print(fbvec)

# Load image
img = nib.load(fimg)
data = img.get_data()
print('data.shape (%d, %d, %d, %d)' %data.shape) # 102 x 102 x 60 x 46 (3D image x 46)

# Load b-values and b-vectors
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)

# Vortex Size
print(img.header.get_zooms()[:3]) # Vortex size = 2 x 2 x 2

# Display image
axial_middle = data.shape[2] // 2 # '//' in python 3 equal to '/' in python 2
plt.figure('Display middle axial slice of DWI')
plt.title('Display middle axial slice of DWI')
plt.axis('off')

# We added transpose because for some reason image looking right
plt.imshow(data[:, :, axial_middle, 0].T, cmap='gray', origin='lower')
plt.savefig('display_middle_axial_slice')
plt.show()

# Gradient Table is just wrapper for b-vals and b-vecs, but it also provides some functionality
gtab = gradient_table(bvals, bvecs)
print(gtab.info)
print(gtab.bvals)
print(gtab.bvecs)

# Extract S0s and save
affine = img.affine
S0s = data[:, :, :, gtab.b0s_mask] # b0s_mask returns index of 0's in b-val
print('S0s.shape (%d, %d, %d, %d)' %S0s.shape) # Usually the 1st b-value is 0, but there can be other 0's as well
for i in range(S0s.shape[3]):
  nib.save(nib.Nifti1Image(S0s, affine), filename + '_S0_' + str(i) + '.nii.gz')

# Display images with background
plt.figure('Remove background 1st 4 DWI')
plt.suptitle('Remove background 1st 4 DWI')
for i in range(4):
  plt.subplot(2, 4, i + 1).set_axis_off()
  plt.imshow(data[:, :, axial_middle, i].T, cmap='gray', origin='lower')

# Remove background
for i in range(4):
  mask, S0_mask = median_otsu(data[:, :, :, i])
  plt.subplot(2, 4, i + 4 + 1).set_axis_off()
  plt.imshow(mask[:, :, axial_middle].T, cmap='gray', origin='lower')
plt.savefig('remove_background')
plt.show()

# Remove background from whole image
# median_radius = 4, num_pass = 4
maskdata, S0_mask = median_otsu(data)
nib.save(nib.Nifti1Image(maskdata, affine), filename + '_mask.nii.gz')