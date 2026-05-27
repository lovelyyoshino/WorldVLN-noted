import numpy as np
import functools


def rotation_to_euler(M, cy_thresh=None, seq='zyx'):
    """
        参考来源：http://afni.nimh.nih.gov/pub/dist/src/pkundu/meica.libs/nibabel/eulerangles.py
        说明：从 3x3 矩阵中求出 Euler 角向量。
        说明：使用上文约定的旋转顺序。
        说明：参数
        ----------
        公式/形状说明：M : array-like, shape (3,3)
        说明：cy_thresh : None 或标量，可选。
        说明：低于该阈值时，不再用直接 arctan 方式估计 x 轴旋转；
        说明：如果为 None（默认），则根据输入精度估计。
        说明：返回值
        -------
        说明：z : scalar
        说明：y : scalar
        说明：x : scalar
        说明：分别表示绕 z、y、x 轴的旋转角，单位为弧度。
        说明：备注
        -----
        说明：如果没有数值误差，该例程可由 Sympy 中先 z、再 y、再 x 的旋转矩阵表达式推导：
        公式/形状说明：[                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
        公式/形状说明：[cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
        公式/形状说明：[sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]
        说明：由此可直接推导 z、y、x：
        公式/形状说明：z = atan2(-r12, r11)
        公式/形状说明：y = asin(r13)
        公式/形状说明：x = atan2(-r23, r33)
        说明：对于 x、y、z 顺序：
        公式/形状说明：y = asin(-r31)
        公式/形状说明：x = atan2(r32, r33)
        公式/形状说明：z = atan2(r21, r11)
        说明：当 cos(y) 接近 0 时会出现问题，因为下面两项都会接近 atan2(0, 0)，数值很不稳定：
        公式/形状说明：z = atan2(cos(y)*sin(z), cos(y)*cos(z))
        公式/形状说明：x = atan2(cos(y)*sin(x), cos(x)*cos(y))
        说明：下面用于缓解数值不稳定的 ``cy`` 修正来自 *Graphics Gems IV*，
        说明：Paul Heckbert（编辑），Academic Press，1994，ISBN: 0123361559。
        说明：具体来自 Ken Shoemake 的 EulerAngles.c，用于处理 cos(y) 接近 0 的情况。
        说明：参见：http://www.graphicsgems.org/
        说明：网站声明该代码“可不受限制地使用”。

    """
    M = np.asarray(M)
    if cy_thresh is None:
        try:
            cy_thresh = np.finfo(M.dtype).eps * 4
        except ValueError:
            cy_thresh = np.finfo(float).eps * 4.0  # 中文说明：_FLOAT_EPS_4
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    # 代码/形状说明：cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = np.sqrt(r33 * r33 + r23 * r23)
    if seq == 'zyx':
        if cy > cy_thresh:  # 中文说明：cos(y) 未接近 0，使用标准形式。
            z = np.arctan2(-r12, r11)  # 代码/形状说明：atan2(cos(y)*sin(z), cos(y)*cos(z))
            y = np.arctan2(r13, cy)  # 中文说明：atan2(sin(y), cy)
            x = np.arctan2(-r23, r33)  # 代码/形状说明：atan2(cos(y)*sin(x), cos(x)*cos(y))
        else:  # 中文说明：cos(y) 接近 0，因此 x -> 0.0，见上文说明。
            # 因此 r21 -> sin(z)，r22 -> cos(z)。
            z = np.arctan2(r21, r22)
            y = np.arctan2(r13, cy)  # 中文说明：atan2(sin(y), cy)
            x = 0.0
    elif seq == 'xyz':
        if cy > cy_thresh:
            y = np.arctan2(-r31, cy)
            x = np.arctan2(r32, r33)
            z = np.arctan2(r21, r11)
        else:
            z = 0.0
            if r31 < 0:
                y = np.pi / 2
                x = np.arctan2(r12, r13)
            else:
                y = -np.pi / 2
    else:
        raise Exception('Sequence not recognized')
    return [z, y, x]


def euler_to_rotation(z=0, y=0, x=0, isRadian=True, seq='zyx'):
    """ 返回说明：返回绕 z、y、x 轴旋转的矩阵。
        说明：使用上文先 z、再 y、再 x 的约定。
        说明：参数
        ----------
        说明：z : scalar
             说明：绕 z 轴的旋转角，单位为弧度（最先执行）。
        说明：y : scalar
             说明：绕 y 轴的旋转角，单位为弧度。
        说明：x : scalar
             说明：绕 x 轴的旋转角，单位为弧度（最后执行）。
        说明：返回值
        -------
        公式/形状说明：M : array shape (3,3)
             说明：与给定角度表示相同旋转的旋转矩阵。
        说明：示例
        --------
        公式/形状说明：>>> zrot = 1.3 # radians
        公式/形状说明：>>> yrot = -0.1
        公式/形状说明：>>> xrot = 0.2
        公式/形状说明：>>> M = euler2mat(zrot, yrot, xrot)
        公式/形状说明：>>> M.shape == (3, 3)
        说明：True
        说明：输出旋转矩阵等于各个单轴旋转矩阵的组合。
        公式/形状说明：>>> M1 = euler2mat(zrot)
        公式/形状说明：>>> M2 = euler2mat(0, yrot)
        公式/形状说明：>>> M3 = euler2mat(0, 0, xrot)
        公式/形状说明：>>> composed_M = np.dot(M3, np.dot(M2, M1))
        公式/形状说明：>>> np.allclose(M, composed_M)
        说明：True
        说明：也可以通过命名参数指定旋转角。
        公式/形状说明：>>> np.all(M3 == euler2mat(x=xrot))
        说明：True
        说明：将 M 应用于向量时，向量应作为列向量放在 M 的右侧。
        说明：如果右侧是 2D 数组而不是单个向量，则该数组每一列代表一个向量。
        公式/形状说明：>>> vec = np.array([1, 0, 0]).reshape((3,1))
        公式/形状说明：>>> v2 = np.dot(M, vec)
        公式/形状说明：>>> vecs = np.array([[1, 0, 0],[0, 1, 0]]).T # 得到 3x2 数组
        公式/形状说明：>>> vecs2 = np.dot(M, vecs)
        说明：旋转方向为逆时针。
        公式/形状说明：>>> zred = np.dot(euler2mat(z=np.pi/2), np.eye(3))
        公式/形状说明：>>> np.allclose(zred, [[0, -1, 0],[1, 0, 0], [0, 0, 1]])
        说明：True
        公式/形状说明：>>> yred = np.dot(euler2mat(y=np.pi/2), np.eye(3))
        公式/形状说明：>>> np.allclose(yred, [[0, 0, 1],[0, 1, 0], [-1, 0, 0]])
        说明：True
        公式/形状说明：>>> xred = np.dot(euler2mat(x=np.pi/2), np.eye(3))
        公式/形状说明：>>> np.allclose(xred, [[1, 0, 0],[0, 0, -1], [0, 1, 0]])
        说明：True
        说明：备注
        -----
        说明：旋转方向由右手定则给出：右手拇指沿旋转轴指向正方向，
        说明：其余手指弯曲的方向就是旋转方向。因此，从旋转轴正向看向负向时，
        说明：这些旋转表现为逆时针。

    """

    if seq != 'xyz' and seq != 'zyx':
        raise Exception('Sequence not recognized')

    if not isRadian:
        z = ((np.pi) / 180.) * z
        y = ((np.pi) / 180.) * y
        x = ((np.pi) / 180.) * x
    if z < -np.pi:
        while z < -np.pi:
            z += 2 * np.pi
    if z > np.pi:
        while z > np.pi:
            z -= 2 * np.pi
    if y < -np.pi:
        while y < -np.pi:
            y += 2 * np.pi
    if y > np.pi:
        while y > np.pi:
            y -= 2 * np.pi
    if x < -np.pi:
        while x < -np.pi:
            x += 2 * np.pi
    if x > np.pi:
        while x > np.pi:
            x -= 2 * np.pi
    assert z >= (-np.pi) and z < np.pi, 'Inappropriate z: %f' % z
    assert y >= (-np.pi) and y < np.pi, 'Inappropriate y: %f' % y
    assert x >= (-np.pi) and x < np.pi, 'Inappropriate x: %f' % x

    Ms = []

    if seq == 'zyx':

        if z:
            cosz = np.cos(z)
            sinz = np.sin(z)
            Ms.append(np.array(
                [[cosz, -sinz, 0],
                 [sinz, cosz, 0],
                 [0, 0, 1]]))
        if y:
            cosy = np.cos(y)
            siny = np.sin(y)
            Ms.append(np.array(
                [[cosy, 0, siny],
                 [0, 1, 0],
                 [-siny, 0, cosy]]))
        if x:
            cosx = np.cos(x)
            sinx = np.sin(x)
            Ms.append(np.array(
                [[1, 0, 0],
                 [0, cosx, -sinx],
                 [0, sinx, cosx]]))
        if Ms:
            return functools.reduce(np.dot, Ms[::-1])
        return np.eye(3)

    elif seq == 'xyz':

        if x:
            cosx = np.cos(x)
            sinx = np.sin(x)
            Ms.append(np.array(
                [[1, 0, 0],
                 [0, cosx, -sinx],
                 [0, sinx, cosx]]))
        if y:
            cosy = np.cos(y)
            siny = np.sin(y)
            Ms.append(np.array(
                [[cosy, 0, siny],
                 [0, 1, 0],
                 [-siny, 0, cosy]]))
        if z:
            cosz = np.cos(z)
            sinz = np.sin(z)
            Ms.append(np.array(
                [[cosz, -sinz, 0],
                 [sinz, cosz, 0],
                 [0, 0, 1]]))

        if Ms:
            return functools.reduce(np.dot, Ms[::-1])
        return np.eye(3)
