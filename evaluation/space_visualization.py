import trimesh
from scipy.spatial import ConvexHull
import numpy as np
space_points = np.load("space_points.npy")
print("num points", space_points.shape[0])
hull = ConvexHull(space_points)
# 假设你已经有了凸包的顶点和面信息
vertices = hull.vertices

# 创建一个trimesh对象
mesh = trimesh.Trimesh(vertices=hull.points, faces=hull.simplices)

# 导出为STL或DAE文件
mesh.export('convex_hull.stl', file_type='stl')
# 或者
# mesh.export('convex_hull.dae', file_type='dae')