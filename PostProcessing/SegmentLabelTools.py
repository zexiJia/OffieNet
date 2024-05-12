import torch
from torch.nn.functional import pad


class SegmentLabelTools:

    def __init__(self, threhold=0.5, strategy='area', scale=None, area_threhold=None):

        assert strategy in ['area', 'threhold'], print(f'strategy 必选在 area 或者 threhold 中选择')    
        self.strategy = strategy
        self.threhold = threhold

        if self.strategy == 'threhold':
            self.area_threhold = area_threhold
            self.method = self.change_via_threhold
            self.scale = scale
            assert self.scale is not None and self.area_threhold is not None, print('在 strategy 为 threhold 模式下 model 会放缩几倍分割图 scale 以及选定面积阈值 threhold 不能为空')
        else:
            self.method = self.change_via_size
        
    
    def change_via_threhold(self, seg_pred):

        mx = torch.max(seg_pred, dim=1, keepdim=True)[0].data 
        mx_size = torch.count_nonzero((seg_pred == mx), dim=1).reshape(mx.shape)

        modi_sign = torch.where(mx_size * self.scale >= self.threhold, 1, -1)
        modi_canidiate = mx * modi_sign

        seg_pred[seg_pred == modi_canidiate] = 1
        seg_pred[seg_pred == -modi_canidiate] = 0

        return seg_pred


    def change_via_size(self, seg_pred):
        
        mx = torch.max(seg_pred, dim=1, keepdim=True)[0].data  # 对比max 与 min 的值是否相等，若不相等，比较面积将面积小的覆盖
        seg_pred[seg_pred == 0] = 1e7
        mi = torch.min(seg_pred, dim=1, keepdim=True)[0].data  # [0].data value
        seg_pred[seg_pred == 1e7] = 0

        # 计算出面积与到中心的距离(假设分割是个正圆), 即：有多少这个数的值，按照 每个 batch 为一个图片
        mx_size = torch.count_nonzero((seg_pred == mx), dim=1)
        mi_size = torch.count_nonzero((seg_pred == mi), dim=1)
        mi_size = mi_size.reshape(mi.shape) # count_nonzero 会降维
        mx_size = mx_size.reshape(mx.shape)
        distance_mx = self._centre_distance(mx, mx_size, (self.temp_w, self.temp_h))
        distance_mi = self._centre_distance(mi, mi_size,  (self.temp_w, self.temp_h))

        modi_canidiate = torch.where(mi_size - distance_mi <= mx_size - distance_mx, mi, mx)  # 将面积小(以及考虑中心)的数值，做出改变的list中的成员
        modi_canidiate = torch.where(mi == mx, -1, modi_canidiate)  # 相等的不改变

        seg_pred[seg_pred == modi_canidiate] = 0 # 修改

        return seg_pred

    def __call__(self, seg_pred, threhold=None):

        self.threhold = threhold if threhold is not None else self.threhold

        if seg_pred.ndim == 3: seg_pred = torch.unsqueeze(seg_pred, dim=1)
        seg_pred = torch.where(seg_pred <= self.threhold, 0, 1)  # 二值化

        seg_pred = pad(seg_pred, (1, 1, 1, 1))  # 在寻找领域的时候，防止溢出
        idx = torch.nonzero(seg_pred, as_tuple=True)
        batch, channels, w, h = seg_pred.shape
        self.temp_w, self.temp_h = w, h
        seg_pred[idx[0], idx[1], idx[2], idx[3]] = idx[2] * w + idx[3]  # 初始化为自己的 indx

         # ---------------------link-----------------------------------
        while(True):

            s = torch.sum(seg_pred)  # 未改变之前的sum值

            self._negbor_to_min(seg_pred, idx, 1, 0)
            self._negbor_to_min(seg_pred, idx, -1, 0)
            self._negbor_to_min(seg_pred, idx, 0, 1)
            self._negbor_to_min(seg_pred, idx, 0, -1)

            if s == torch.sum(seg_pred):  # 如果 sum 值不再改变，达到平衡
                break

        # ---------------------- mask-------------------------------------
        
        seg_pred = seg_pred.reshape(batch, -1)
        while(True):

            number_nonzero = torch.count_nonzero(seg_pred)  # 初始非零元素的个数，若不再改变，代表终止

            seg_pred = self.method(seg_pred)
        
            if torch.count_nonzero(seg_pred) == number_nonzero: break  # 未发生改变, break

        seg_pred[seg_pred != 0] = 1
        seg_pred = seg_pred.reshape(batch, channels, self.temp_w, self.temp_h)
        # print(seg_pred.shape)

        return seg_pred[:, :, 1:-1, 1:-1]


    def _centre_distance(self, value, area, shape):

        w, h = shape
        x = w / 2
        y = h / 2
        vy = value % w
        vx = (value - vy) / w

        r = torch.sqrt(2 * area / 3.141592654)
        vx += r

        return 2 * ((vx - x) ** 2 + (vy - y) ** 2)


    def _negbor_to_min(self, seg_pred, seg_pred_non_zero_idx, detla_x=0, detla_y=0):

        idx = seg_pred_non_zero_idx  # 通过non_zero 的到 4 triple 值

        seg_pred_non_zero = seg_pred[idx[0], idx[1], idx[2], idx[3]]
        a1 = seg_pred[idx[0], idx[1], idx[2] + detla_x,
                    idx[3]+detla_y]     # 4周围的邻域，是否也是非零，若非零则比较大小，赋予最小的值
        nidx = torch.nonzero(a1, as_tuple=True)

        seg_pred[idx[0][nidx[0]], idx[1][nidx[0]], idx[2][nidx[0]], idx[3][nidx[0]]] = torch.where(
            seg_pred_non_zero[nidx[0]] < a1[nidx[0]], seg_pred_non_zero[nidx[0]], a1[nidx[0]])





if __name__ == '__main__':

    import numpy as np
    from matplotlib import pyplot as plt
    a = np.load('c.npy')

    np.set_printoptions(threshold=np.inf, linewidth=700)
    torch.set_printoptions(edgeitems=512, linewidth=700)
    plt.subplot(121)
    plt.imshow(np.squeeze(a))

    # a = torch.ones((2, 1, 4, 5), dtype=torch.long)
    # a[0, :, 2,  :] = 0
    # a[1, :, 3,  :] = 0

    # print(a)

    seg = label(torch.tensor(a), 0.5)
    a[a >= 0.5] = 1
    plt.subplot(122)
    plt.imshow(np.squeeze(seg.numpy()))
    plt.savefig('a.png')
    # print(torch.count_nonzero(seg), np.count_nonzero(a>=0.5))
