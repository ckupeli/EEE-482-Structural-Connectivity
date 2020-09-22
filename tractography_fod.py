import numpy as np
from os.path import join
import nibabel as nib
from dipy.io.streamline import load_tck
from dipy.segment.quickbundles import QuickBundles
from dipy.viz import window, actor
from dipy.io.pickles import save_pickle

filename = 'DTI30s010'

# Load streamlines
streamlines, header = load_tck(filename + '_fod_streamlines.tck')
print('Number of streamlines %i' %len(streamlines))

# Do steamline clustering using QuickBundles (QB) using FOD (Probalistic Method)
# dist_thr (distance threshold) which affects number of clusters and their size
# pts (number of points in each streamline) which will be used for downsampling before clustering
# Default values : dist_thr = 4 & pts = 12
qb = QuickBundles(streamlines, dist_thr=42, pts=18)
clusters = qb.clusters()
print('Number of clusters %i' %qb.total_clusters)
print('Cluster size', qb.clusters_sizes())

# Display streamlines
ren = window.Renderer()
ren.add(actor.streamtube(streamlines, window.colors.white))
ren.set_camera(position=(-176.42, 118.52, 128.20),
                    focal_point=(113.30, 128.31, 76.56),
                    view_up=(0.18, 0.00, 0.98))
window.show(ren)
window.record(ren, out_path=filename + '_stream_lines_fod.png', size=(600, 600))

# Display centroids
window.clear(ren)
colormap = actor.create_colormap(np.arange(qb.total_clusters))
ren.add(actor.streamtube(streamlines, window.colors.white, opacity=0.1))
ren.add(actor.streamtube(qb.centroids, colormap, linewidth=0.5))
ren.set_camera(position=(-176.42, 118.52, 128.20),
                    focal_point=(113.30, 128.31, 76.56),
                    view_up=(0.18, 0.00, 0.98))
window.show(ren)
window.record(ren, out_path=filename + '_centroids_fod.png', size=(600, 600))

# Display tracks
window.clear(ren)
colormap_full = np.ones((len(streamlines), 3))
for cluster, color in zip(clusters.items(), colormap):
  colormap_full[cluster[1]['indices']] = color
ren.add(actor.streamtube(streamlines, colormap_full))
ren.set_camera(position=(-176.42, 118.52, 128.20),
                    focal_point=(113.30, 128.31, 76.56),
                    view_up=(0.18, 0.00, 0.98))
window.show(ren)
window.record(ren, out_path=filename + '_stream_line_cluster_fod.png', size=(600, 600))
save_pickle(filename + '_qb_fod.pkl', clusters)