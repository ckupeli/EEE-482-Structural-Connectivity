import numpy as np
from os.path import join
import nibabel as nib
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel, fractional_anisotropy, quantize_evecs
from dipy.data import get_sphere
from dipy.tracking.eudx import EuDX
from dipy.segment.quickbundles import QuickBundles
from dipy.viz import window, actor
from dipy.io.streamline import save_trk
from dipy.io.pickles import save_pickle

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

# Compute FA
fa = fractional_anisotropy(tenfit.evals)

# Compute spherical coordinates
sphere = get_sphere('symmetric724') # Symmetric Sphere with 724 vertices

# Get quantize vectors
ind = quantize_evecs(tenfit.evecs, sphere.vertices)
print('ind.shape (%d, %d, %d)' %ind.shape)

# Compute Eular Delta Crossing with FA
eu = EuDX(a=fa, ind=ind, seeds=100000, odf_vertices=sphere.vertices, a_low=0.2) # FA uses a_low = 0.2
streamlines = [line for line in eu]
print('Number of streamlines %i' %len(streamlines))
'''
for line in streamlines:
  print(line.shape)
'''
# Do steamline clustering using QuickBundles (QB) using Eular's Method
# dist_thr (distance threshold) which affects number of clusters and their size
# pts (number of points in each streamline) which will be used for downsampling before clustering
# Default values : dist_thr = 4 & pts = 12
qb = QuickBundles(streamlines, dist_thr=20, pts=20)
clusters = qb.clusters()
print('Number of clusters %i' %qb.total_clusters)
print('Cluster size', qb.clusters_sizes())

# Display streamlines
ren = window.Renderer()
ren.add(actor.streamtube(streamlines, window.colors.white))
window.show(ren)
window.record(ren, out_path=filename + '_stream_lines_eu.png', size=(600, 600))

# Display centroids
window.clear(ren)
colormap = actor.create_colormap(np.arange(qb.total_clusters))
ren.add(actor.streamtube(streamlines, window.colors.white, opacity=0.1))
ren.add(actor.streamtube(qb.centroids, colormap, linewidth=0.5))
window.show(ren)
window.record(ren, out_path=filename + '_centroids_eu.png', size=(600, 600))

# Display tracks
window.clear(ren)
colormap_full = np.ones((len(streamlines), 3))
for cluster, color in zip(clusters.items(), colormap):
  colormap_full[cluster[1]['indices']] = color
ren.add(actor.streamtube(streamlines, colormap_full))
window.show(ren)
window.record(ren, out_path=filename + '_stream_line_cluster_eu.png', size=(600, 600))

# Save Streamline files
save_trk(filename + "_stream_line_eu.trk", streamlines=streamlines, affine=np.eye(4))
save_pickle(filename + '_qb_eu.pkl', clusters)