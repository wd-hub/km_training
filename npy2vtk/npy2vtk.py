# change miniconda enviroment to the default python 2.7 enviroment

import numpy as np
from numpy_support import numpy_to_vtk, vtk_to_numpy
import vtk
from pyevtk.hl import imageToVTK, pointsToVTK

path = '/home/dong/Documents/3D_Matching/depth_patch_training/pcl_data/data/fr1_desk/1305031453.374112.npy'
data = np.load(path, encoding='latin1')
NumPy_data_shape = data.shape

VTK_data = numpy_to_vtk(num_array=data.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
imageToVTK("./image", cellData = {"temp": data}, pointData = {"temp" : data})

NumPy_data = vtk_to_numpy(VTK_data)
NumPy_data = NumPy_data.reshape(NumPy_data_shape)

print('Done!')