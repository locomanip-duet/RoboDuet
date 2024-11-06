import cv2
import numpy as np
from apriltag import apriltag
import time
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

def rotation_matrix_to_quaternion(R):
    """
    Converts a 3x3 rotation matrix to a quaternion.

    :param R: 3x3 rotation matrix
    :return: Quaternion [w, x, y, z]
    """
    trace = np.trace(R)
    if trace > 0:
        S = 2.0 * np.sqrt(1.0 + trace)
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    return np.array([qx, qy, qz, qw])

def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = np.cross(xyz, b, axis=-1) * 2
    result = (b + a[:, 3:] * t + np.cross(xyz, t, axis=-1)).reshape(shape)
    return result

def quat_to_angle(quat):
    y_vector = np.array([0., 1., 0.])
    z_vector = np.array([0., 0., 1.])
    x_vector = np.array([1., 0., 0.])
    roll_vec = quat_apply(quat, x_vector)  # [0,1,0]
    roll = np.arctan2(roll_vec[1], roll_vec[0])  # roll angle = arctan2(y, x)
    pitch_vec = quat_apply(quat, y_vector)  # [0,0,1]
    pitch = -np.arctan2(pitch_vec[2], pitch_vec[1])  # pitch angle = arctan2(z, y)
    yaw_vec = quat_apply(quat, z_vector)  # [1,0,0]
    yaw = -np.arctan2(yaw_vec[0], yaw_vec[2])  # yaw angle = arctan2(x, z)

    return np.stack([roll, pitch, yaw], axis=-1)


def rectify_translation(v):
    x, y, z = v[0], v[1], v[2]
    x_new = z
    y_new = -x
    z_new = -y
    return np.array([x_new, y_new, z_new])




class RealTimeCamera:
    def __init__(self, camera_id=0, tag_type="tag36h11") -> None:
        self.capture = cv2.VideoCapture(camera_id)
        self.detector = apriltag(tag_type)

        # realsense D435i K intrinsic params
        self.camera_matrix = np.array([908.6600820265805, 0.0, 644.5248851267536, 0.0, 907.9504674173401, 367.762927245881, 0.0, 0.0, 1.0], dtype=np.float32).reshape((3,3))
        # distortion coefficients
        self.dis_coeffs = np.array([0.08610810269384651, -0.1530313478779539, 0.0017606565123906837, -0.0018913721543482396, 0.0], dtype=np.float32)

    def getImage(self):
        ret, frame = self.capture.read()
        if not ret:
            raise ValueError("Failed to capture image from camera")
        return frame
    
    def detect(self):
        image = self.getImage()
        detections = detector.detect(image)
            
        tag_side_length_half = 0.15 / 2 # (m) x right, y down, z vertical from eyes
        object_points = np.array([[-tag_side_length_half, tag_side_length_half, 0],
                                [tag_side_length_half, tag_side_length_half, 0],
                                [tag_side_length_half, -tag_side_length_half, 0],
                                [-tag_side_length_half, -tag_side_length_half, 0]], dtype=np.float32)
        image_points = detections[0]['lb-rb-rt-lt']

        retval, rvec, tvec = cv2.solvePnP(object_points, image_points, self.camera_matrix, self.dis_coeffs)
        rotation_matrix, _ = cv2.Rodrigues(rvec) # angle vec to mat
        angles_in_degrees = rvec * (180.0 / np.pi)
        rvec_flat, tvec_flat = rvec.flatten(), tvec.flatten()
        quat = rotation_matrix_to_quaternion(rotation_matrix)
        abg = quat_to_angle(quat)
        tvec_rectify = rectify_translation(tvec_flat)
        
        # print("quat: ", quat)
        # print("abg: ", abg)
        # print("tvec_rectify: ", tvec_rectify)
        return np.concatenate([tvec_rectify, abg], axis=-1)
    
if __name__ == '__main__':
    imagepath = 'real.png'
    image = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
    detector = apriltag("tag36h11")
    detections = detector.detect(image)

    # realsense D435i K intrinsic params
    camera_matrix = np.array([908.6600820265805, 0.0, 644.5248851267536, 0.0, 907.9504674173401, 367.762927245881, 0.0, 0.0, 1.0], dtype=np.float32).reshape((3,3))
    # distortion coefficients
    dis_coeffs = np.array([0.08610810269384651, -0.1530313478779539, 0.0017606565123906837, -0.0018913721543482396, 0.0], dtype=np.float32)
    # world coordinates of selected points, O is the center of AprilTag
    tag_side_length_half = 0.15 / 2 # (m) x right, y down, z vertical from eyes
    object_points = np.array([[-tag_side_length_half, tag_side_length_half, 0],
                              [tag_side_length_half, tag_side_length_half, 0],
                            [tag_side_length_half, -tag_side_length_half, 0],
                            [-tag_side_length_half, -tag_side_length_half, 0]], dtype=np.float32)
    image_points = detections[0]['lb-rb-rt-lt']

    retval, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dis_coeffs)
    rotation_matrix, _ = cv2.Rodrigues(rvec) # angle vec to mat
    angles_in_degrees = rvec * (180.0 / np.pi)


    # visualize 3D graph
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    axes = np.eye(3, dtype=np.float64)
    rvec_flat, tvec_flat = rvec.flatten(), tvec.flatten()
    quat = rotation_matrix_to_quaternion(rotation_matrix)
    abg = quat_to_angle(quat)
    tvec_rectify = rectify_translation(tvec_flat)
    print("quat: ", quat)
    print("abg: ", abg)
    print("tvec_rectify: ", tvec_rectify)

    # Here we create the arrows:
    arrow_prop_dict = dict(mutation_scale=20, arrowstyle='->', shrinkA=0, shrinkB=0)
    cam_base = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]
                          ]).reshape((3, 3))

    rotation_matrix = np.dot(rotation_matrix, cam_base)

    R_for_visual = np.array([1, 0, 0, 0, 0, 1, 0, -1, 0]).reshape((3, 3))
    rotation_matrix = np.dot(R_for_visual, rotation_matrix)
    cam_base = np.dot(R_for_visual, cam_base)

    a = Arrow3D([tvec_flat[0], tvec_flat[0] + rotation_matrix[0][0]], [tvec_flat[1], tvec_flat[1] + rotation_matrix[1][0]], [tvec_flat[2], tvec_flat[2] + rotation_matrix[2][0]], **arrow_prop_dict, color='r')
    ax.add_artist(a)
    a = Arrow3D([tvec_flat[0], tvec_flat[0] + rotation_matrix[0][1]], [tvec_flat[1], tvec_flat[1] + rotation_matrix[1][1]], [tvec_flat[2], tvec_flat[2] + rotation_matrix[2][1]], **arrow_prop_dict, color='g')
    ax.add_artist(a)
    a = Arrow3D([tvec_flat[0], tvec_flat[0] + rotation_matrix[0][2]], [tvec_flat[1], tvec_flat[1] + rotation_matrix[1][2]], [tvec_flat[2], tvec_flat[2] + rotation_matrix[2][2]], **arrow_prop_dict, color='b')
    ax.add_artist(a)
    
    # a = Arrow3D([0, 0 + rotation_matrix[0][0]], [0, 0 + rotation_matrix[1][0]], [0, 0 + rotation_matrix[2][0]], **arrow_prop_dict, color='r')
    # ax.add_artist(a)
    # a = Arrow3D([0, 0 + rotation_matrix[0][1]], [0, 0 + rotation_matrix[1][1]], [0, 0 + rotation_matrix[2][1]], **arrow_prop_dict, color='g')
    # ax.add_artist(a)
    # a = Arrow3D([0, 0 + rotation_matrix[0][2]], [0, 0 + rotation_matrix[1][2]], [0, 0 + rotation_matrix[2][2]], **arrow_prop_dict, color='b')
    # ax.add_artist(a)

    a = Arrow3D([0, 0 + cam_base[0][0]], [0, 0 + cam_base[1][0]], [0, 0 + cam_base[2][0]], **arrow_prop_dict, color='r')
    ax.add_artist(a)
    a = Arrow3D([0, 0 + cam_base[0][1]], [0, 0 + cam_base[1][1]], [0, 0 + cam_base[2][1]], **arrow_prop_dict, color='g')
    ax.add_artist(a)
    a = Arrow3D([0, 0 + cam_base[0][2]], [0, 0 + cam_base[1][2]], [0, 0 + cam_base[2][2]], **arrow_prop_dict, color='b')
    ax.add_artist(a)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('6DoF of AprilTag')

    plt.savefig("1_res_official_3d.png")