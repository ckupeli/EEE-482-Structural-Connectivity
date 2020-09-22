import numpy as np
from os.path import join
import nibabel as nib
from dipy.io import read_bvals_bvecs, utils
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel, mean_diffusivity, fractional_anisotropy, color_fa
import matplotlib.pyplot as plt

filename = 'DTI30s010'

# Get directory of DWI
fimg = join('./', filename + '_mask.nii.gz') # 4D NIfTI1 file
print(fimg)
fbval = join('./', filename + '.bval') # b-values
print(fbval)
fbvec = join('./', filename + '.bvec') # b-vectors
print(fbvec)

# Load image
img = nib.load(fimg)
affine = img.affine
data = img.get_data() # 102 x 102 x 60 x 46 (3D image x 46)

# Load b-values and b-vectors
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
gtab = gradient_table(bvals, bvecs)

# Create tensor model
tenmodel = TensorModel(gtab)

# Fit the data using created tensor model
tenfit = tenmodel.fit(data)

# Compute fractional anisotropy (FA)
fa = fractional_anisotropy(tenfit.evals)
fa[np.isnan(fa)] = 0 # Currently generated data does not contain nan values, but it might cause some problem if we try using another dataset which can generates nan's
img_fa = nib.Nifti1Image(fa, affine)
data_fa = img_fa.get_data()
print('data_fa.shape (%d, %d, %d)' %data_fa.shape) # 102 x 102 x 60
nib.save(nib.Nifti1Image(fa.astype(np.float32), affine), filename + '_tensor_fa.nii.gz')
img_evecs = nib.Nifti1Image(tenfit.evecs, affine)
data_evecs = img_evecs.get_data() # Eigen values & eigen vectors of tensor
print('data_evects.shape (%d, %d, %d, %d, %d)' %data_evecs.shape)
nib.save(img_evecs, filename + '_tensor_evecs.nii.gz')

# Display FA
axial_middle = data.shape[2] // 2 # '//' in python 3 equal to '/' in python 2
plt.figure('Display fractional anisotropy')
plt.title('Display fractional anisotropy')
plt.axis('off')
plt.imshow(data_fa[:, :, axial_middle].T, cmap='gray', origin='lower')
plt.savefig('display_fa')
plt.show()

# Compute mean diffusivity (MD)
md = mean_diffusivity(tenfit.evals)
img_md = nib.Nifti1Image(md.astype(np.float32), affine)
data_md = img_md.get_data()
nib.save(img_md, filename + '_tensors_md.nii.gz')

# Display MD
plt.figure('Display mean diffusivity')
plt.title('Display mean diffusivity')
plt.axis('off')
plt.imshow(data_md[:, :, axial_middle].T, cmap='gray', origin='lower')
plt.savefig('display_md')
plt.show()

# Compute RGB FA
fa = np.clip(fa, 0, 1) # Scale fa between 0 and 1
rgb = color_fa(fa, tenfit.evecs)
img_rgb = nib.Nifti1Image(255 * rgb, img.affine)
data_rgb = img_rgb.get_data()
print('data_rgb.shape (%d, %d, %d, %d)' %data_rgb.shape)
nib.save(img_rgb, filename + '_tensor_rgb.nii.gz')

# Use 3rd party software to display better images for the report