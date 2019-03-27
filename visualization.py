import numpy as np
import io
import os
from itertools import product, combinations
import matplotlib
matplotlib.use("Agg")
import PIL.Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torchvision.transforms import ToTensor
import imageio

hparts = ["Spine",
"LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase", 
"RightUpLeg",  "RightLeg", "RightFoot", "RightToeBase", 
'Spine1', 'Spine2', 'Neck', 'Head',
"LeftShoulder",  "LeftArm", "LeftForeArm", "LeftHand",
"RightShoulder", "RightArm", "RightForeArm", "RightHand"]

hfathers = ['Spine', 
'Spine', 'LeftUpLeg', 'LeftLeg', 'LeftFoot',
'Spine', 'RightUpLeg', 'RightLeg', 'RightFoot',
'Spine', 'Spine1', 'Spine2', 'Neck',
'Spine2', "LeftShoulder",  "LeftArm", "LeftForeArm",
'Spine2', "RightShoulder", "RightArm", "RightForeArm"]

pparts = ["Spine", 
"LeftUpLeg", "LeftFoot", "LeftToeBase",
"RightUpLeg", "RightFoot", "RightToeBase",
"Spine1", "Spine2", "Neck", "Head",
"LeftShoulder", "LeftHand",
"RightShoulder", "RightHand",
]

pfathers = ['Spine',
'Spine', 'LeftUpLeg', 'LeftFoot',
'Spine', 'RightUpLeg', 'RightFoot',
'Spine', 'Spine1', 'Spine2', 'Neck',
'Spine2', "LeftShoulder", 
'Spine2', "RightShoulder"]

human_limbs = [(hparts.index(j[0]), hparts.index(j[1])) for j in zip(hparts, hfathers)]
penguin_limbs = [(pparts.index(j[0]), pparts.index(j[1])) for j in zip(pparts, pfathers)]


def plot_motion(motion3d, phase="h"):
    # plot pose to visualize in training
    # only draw the middle frame
    length = motion3d.shape[-1]
    joints3d = motion3d[:, :, length // 2]
    buf = io.BytesIO()
    draw_skel(joints3d, save_buf=buf, phase=phase)
    img = PIL.Image.open(buf)
    img = ToTensor()(img).unsqueeze(0)
    return img


def draw_skel(joints3d_origin, save_path=None, phase="h", is_show=False, save_buf=None):
    """
        joints3d : shape (n_joints, 3)
    """
    if phase == "h":
        limbs = human_limbs
    else:
        limbs = penguin_limbs
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    joints3d = joints3d_origin.copy()
    avg_point = np.mean(joints3d, axis=0)

    joints3d -= avg_point
    p, c = joints3d.shape
    '''
    if not(p == 21 and c == 3):
        raise Exception('Except the input array in draw_skel() has shape (21, 3)')
        '''
    settings = np.arange(p)
    colors = settings / p
    remain_id = [l[0] for l in limbs]
    ax.scatter(joints3d[remain_id, 0], joints3d[remain_id, 2], joints3d[remain_id, 1], c=colors)
    for c, f in limbs:
        cx, cy, cz = joints3d[c]
        fx, fy, fz = joints3d[f]
        ax.plot([cx, fx], [cz, fz], [cy, fy], c='black')
    '''
    for i, (x, y, z) in enumerate(joints3d):
        direction = 1 if i % 2 == 0 else -1
        offset = 5 * direction
        ax.text(-x - offset, z + offset, y - offset, s = str(i))
    '''
    t = np.max(joints3d)
    ax.set_aspect('equal')
    ax.set_xlim(-t + 0.1, t + 0.1)
    ax.set_ylim(-t + 0.1, t + 0.1)
    ax.set_zlim(-t + 0.1, t + 0.1)

    if is_show:
        plt.show()
    if save_buf is not None:
        plt.savefig(save_buf, format="jpeg")
        save_buf.seek(0)
    if save_path is not None:
        plt.savefig(save_path, transparent=True)
    plt.close()


def draw_skel2d(joints3d_origin, ax_lim = None):
    """
        joints3d : shape (n_joints, 3)
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    joints3d = joints3d_origin.copy()
    avg_point = np.mean(joints3d, axis=0)

    joints3d -= avg_point
    p, c = joints3d.shape
    limbs = None
    if p == 21:
        limbs = human_limbs
    elif p == 15:
        limbs = penguin_limbs
    else:
        raise Exception("Invalid skeleton!")

    settings = np.arange(p)
    colors = settings / p
    ax.scatter(joints3d[:, 0], joints3d[:, 2], joints3d[:, 1], c=colors)
    for c, f in limbs:
        cx, cy, cz = joints3d[c]
        fx, fy, fz = joints3d[f]
        ax.plot([cx, fx], [cz, fz], [cy, fy], c='black')
    '''
    for i, (x, y, z) in enumerate(joints3d):
        direction = 1 if i % 2 == 0 else -1
        offset = direction * 0.5
        ax.text(-x - offset, z + offset, y - offset, s = str(i))
    '''
    t = np.max(joints3d) if ax_lim is None else ax_lim
    ax.set_aspect('equal')
    ax.set_xlim(-t + 0.1, t + 0.1)
    ax.set_ylim(-t + 0.1, t + 0.1)
    ax.set_zlim(-t + 0.1, t + 0.1)

    return fig


def frames2video(motion3d, path):
    p, c, f = motion3d.shape
    images = []

    t = 10
    for i in range(f):
        fig = draw_skel2d(motion3d[:, :, i], ax_lim = t)
        fig.canvas.draw()

        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        images.append(data)
    images = np.array(images)
    save_dir = os.path.dirname(path)

    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    print('saving %s' % path)
    videowriter = imageio.get_writer(path, fps=25)

    for im in images:
        videowriter.append_data(im)
    videowriter.close()


if __name__ == '__main__':
    from transform import generate_penguin
    path = "/data1/wurundi/cmu-mocap/074/01_01/0.npy"
    motion3d = np.load(path)
    motion3d = generate_penguin(motion3d)
    motion3d -= motion3d[5]
    print(motion3d[..., 0])
    draw_skel(motion3d[..., 0], "./test.png", phase="nh")
