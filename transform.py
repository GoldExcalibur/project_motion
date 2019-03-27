import numpy as np
import glob
import os
import copy

unique_index = [0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 15, 16, 18, 19, 20, 22, 25, 26, 27, 29]

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

hparts = hparts
hfathers = hfathers

nparts = len(hparts)
id_dict = dict(zip(hparts, range(nparts)))


class Node(object):
    def __init__(self, own_id, name):
        self.own_id = own_id
        self.name = name
        self.father = None
        self.offset = None
        self.realpos = None

    def set_offset(self, pos, father_pos):
        father_id = self.father.own_id
        if self.own_id != father_id:
            self.offset = pos - father_pos
        else:
            self.offset = pos.copy()
            #         print(self.own_id, pos, father_id, father_pos)

    def set_father(self, father):
        if isinstance(father, Node) == False:
            raise Exception("Argument Type must be Node!")
        self.father = father

    def get_realpos(self):
        father_id = self.father.own_id
        if self.realpos is not None:
            return self.realpos, self.own_id, father_id
        realpos = self.offset.copy()
        if self.own_id != father_id:
            #             print(self.own_id, self.offset, father_id, self.father.get_realpos())
            father_pos, _, _ = self.father.get_realpos()
            realpos += father_pos
        self.realpos = realpos
        return realpos, self.own_id, father_id

    def del_father(self):
        assert self.father != self and self.father != self.father.father
        tmp = self.father
        self.father = tmp.father
        del tmp

    def scale(self, scale):
        self.offset *= scale

    def translate(self, shift):
        self.offset += shift


def construct_motionTree(motion3D, tags=hparts, fathers=hfathers):
    if isinstance(motion3D, np.ndarray) is False:
        raise Exception('motion3D has to be numpy array')
    p, c, f = motion3D.shape
    assert p == nparts and c == 3
    nodelist = []
    for i in range(p):
        name = tags[i]
        # realpos = motion3D[i]
        n = Node(i, name)
        nodelist.append(n)
    for i in range(p):
        name = fathers[i]
        father_id = tags.index(name)
        n = nodelist[i]
        n.set_father(nodelist[father_id])
        n.set_offset(motion3D[i], motion3D[father_id])

    return nodelist


'''
naive transforamtion from human skeleton to penguin skeleton

'''


def penguin_transform(nodelist):
    p = len(nodelist)
    nodelist_cp = copy.deepcopy(nodelist)
    for n in nodelist_cp:
        n.realpos = None
    rarm_id, larm_id = id_dict['RightArm'], id_dict['LeftArm']
    nodelist_cp[rarm_id].scale(1.5)
    nodelist_cp[larm_id].scale(1.5)
    sp1_id, sp2_id = id_dict['Spine1'], id_dict['Spine2']
    nodelist_cp[sp1_id].scale(2.0)
    nodelist_cp[sp2_id].scale(2.0)
    ls_id, rs_id = id_dict['LeftShoulder'], id_dict['RightShoulder']
    nodelist_cp[ls_id].scale(1.2)
    nodelist_cp[rs_id].scale(1.2)
    luleg_id, ruleg_id = id_dict['LeftUpLeg'], id_dict['RightUpLeg']
    nodelist_cp[luleg_id].scale(2.0)
    nodelist_cp[ruleg_id].scale(2.0)
    lleg_id, rleg_id = id_dict['LeftLeg'], id_dict['RightLeg']
    lfoo_id, rfoo_id = id_dict['LeftFoot'], id_dict['RightFoot']
    nodelist_cp[lfoo_id].del_father()
    nodelist_cp[rfoo_id].del_father()
    #     nodelist_cp[lfoo_id].translate(nodelist_cp[luleg_id].get_realpos() - nodelist_cp[lleg_id].get_realpos())
    #     nodelist_cp[rfoo_id].translate(nodelist_cp[ruleg_id].get_realpos() - nodelist_cp[rleg_id].get_realpos())
    nodelist_cp[lfoo_id].scale(0.5)
    nodelist_cp[rfoo_id].scale(0.5)

    erase_id = [id_dict[t] for t in ['RightForeArm', 'LeftForeArm', 'LeftHand', 'RightHand']] + [lleg_id, rleg_id]
    remain_id = set(range(p)) - set(erase_id)
    motion3d = []
    # limbs = []
    for i in remain_id:
        realpos, own_id, father_id = nodelist_cp[i].get_realpos()
        # limbs.append([own_id, father_id])
        motion3d.append(realpos)
    return np.array(motion3d)


def generate_penguin(motion3d):
    nodelist = construct_motionTree(motion3d)
    pmotion3d = penguin_transform(nodelist)
    return pmotion3d


def preprocess(motion3d, subrate=2, window=128):
    p, c, f = motion3d.shape
    print('Preprocess motion3d data with subsample rate %d window sizw %d' % (subrate, window))
    motion3d_without_tpose = motion3d[:, :, range(1, f, subrate)]
    p, c, ff = motion3d_without_tpose.shape
    motion3d_single_list = []
    for s in range(0, ff - window, window // 2):
        motion3d_single = motion3d_without_tpose[:, :, s:s + window]
        motion3d_single_list.append(motion3d_single)
    print('After preprocessing, we get %d single motions' % (len(motion3d_single_list)))
    return motion3d_single_list


def trans_motion3d(motion3d, base_id=0):
    centers = motion3d[base_id, :, :]
    motion_trans = motion3d - centers

    # centers = centers - centers[:, 0].reshape(3, 1)

    # adding velocity
    velocity = np.c_[np.zeros((3, 1)), centers[:, 1:] - centers[:, :-1]].reshape(1, 3, -1)
    motion_trans = np.r_[motion_trans[:base_id], motion_trans[base_id+1:], velocity]
    return motion_trans


def trans_motion3d_inv(motion3d, base_id=0, sx=0, sy=0, sz=0):
    if len(motion3d.shape) == 2:
        motion3d = motion3d.reshape(-1, 3, motion3d.shape[-1])
    velocity = motion3d[-1].copy()
    motion_inv = np.r_[motion3d[:base_id], np.zeros((1, 3, motion3d.shape[-1])), motion3d[base_id:-1]]

    # restore centre position
    centers = np.zeros_like(velocity)
    sum = 0
    for i in range(motion3d.shape[-1]):
        sum += velocity[:, i]
        centers[:, i] = sum
    centers += np.array([[sx], [sy], [sz]])

    return motion_inv + centers.reshape(1, 3, -1)


def lr_flip(motion, phase):
    """Input motion should be the output of trans_motion3d
    """
    if phase == 'h':
        joint_pair = [4, 5, 6, 7, 0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 12, 13, 14, 15, 20]
    else:
        joint_pair = [3, 4, 5, 0, 1, 2, 6, 7, 8, 9, 12, 13, 10, 11, 14]
    # flip_mask = 0
    # unflip_mask = [1, 2]
    flip_motion = motion.copy()
    flip_motion[:, 0] = - motion[joint_pair, 0]
    flip_motion[:, 1] = motion[joint_pair, 1]
    flip_motion[:, 2] = motion[joint_pair, 2]
    return flip_motion


def mean_distance(motion_pred, motion_gt, base_id):
    assert motion_pred.shape == motion_gt.shape
    motion_pred_norm = motion_pred - motion_pred[base_id, :, :1]
    motion_gt_norm = motion_gt - motion_gt[base_id, :, :1]
    dis_2 = np.sum((motion_pred_norm - motion_gt_norm)**2, axis = 1)
    dis_mean = np.mean(np.sqrt(dis_2))
    return dis_mean


if __name__ == '__main__':
    from visualization import draw_skel
    h_motion = np.load("/data1/wurundi/cmu-mocap/074/74_14/0.npy")
    nh_motion = generate_penguin(np.load("/data1/wurundi/cmu-mocap/074/74_14/0.npy"))

    draw_skel(h_motion[:, :, 80], save_path='test_real_A_ori.png', phase='h')
    draw_skel(nh_motion[:, :, 80], save_path='test_real_B_ori.png', phase='nh')

    h_motion_flip = trans_motion3d_inv(lr_flip(trans_motion3d(h_motion, base_id=0), "h"), base_id=0)
    nh_motion_flip = trans_motion3d_inv(lr_flip(trans_motion3d(nh_motion, base_id=0), "nh"), base_id=0)

    draw_skel(h_motion_flip[:, :, 80], save_path='test_real_A_ori_flip.png', phase='h')
    draw_skel(nh_motion_flip[:, :, 80], save_path='test_real_B_ori_flip.png', phase='nh')
