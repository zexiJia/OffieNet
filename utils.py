# --------------------------------------------------------
# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import io
import os
import math
import time
import json
from collections import defaultdict, deque
import datetime
import numpy as np
from timm.utils import get_state_dict

from pathlib import Path

import torch
import torch.distributed as dist
from torch._six import inf

from tensorboardX import SummaryWriter
from glob import glob
from scipy import ndimage

# 保存与读取权值文件
def save_checkpoint(model, optimizer, epoch, dir_base, it=None):
    
    if not os.path.exists(dir_base):
        os.mkdir(dir_base)

    state = {
        'net_weights': model.state_dict(),
        'opt_parameters': optimizer.state_dict(),
        'epoch': epoch
    }
    epoch = epoch if it is None else f'{epoch}_{it}'
    torch.save(state, os.path.join(dir_base, f'{epoch}.pth'))



def load_checkpoint(checkpoint_path, model, optimizer=None, lr_scheduler=None, strict=True):

    checkpoint = torch.load(checkpoint_path)

    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['net_weights'], strict=strict)

    if optimizer is None and lr_scheduler is None:
        return  checkpoint['epoch']
    
    if lr_scheduler is not None:
        optimizer.load_state_dict(checkpoint['opt_parameters'])

    if 'lrs_parameters' in checkpoint and lr_scheduler != None: # 初期没有加这个参数
        lr_scheduler.load_state_dict(checkpoint['lrs_parameters'])
    else:
        print(f'lrs_parameters is not loaded')

    start_epoch = int(checkpoint['epoch']) + 1

    return start_epoch


def InitLogging(dst, logger_name='pretrain'):

    logger_name = logger_name
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())       
    fh = logging.FileHandler(os.path.join(dst, 'experiment.log'))
    # // fh.setFormatter(logging.Formatter('In %(lineno)d of %(filenames)s at %(asctime)s : %(message)s'))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)



def PRC_new(y_true, y_pred, maxo=np.pi/6, maxd=15):
    
    from scipy import spatial

    if y_pred.shape[0] == 0:
        return 0, 0, y_true.shape[0], 0, 0
    if y_true.shape[0] == 0:
        return 0, 0, y_true.shape[0], 0, 0
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    dis = spatial.distance.cdist(y_pred[:, :2], y_true[:, :2], 'euclidean')
    matched_num = (dis <= maxd).sum(axis=1)
    dis = dis[matched_num.argsort(), :]
    y_pred = y_pred[matched_num.argsort(), :]
    y_true2 = np.copy(y_true)
    TP = 0
    for i in range(y_pred.shape[0]):
        if y_true2.shape[0] == 0:
            break
        dis = spatial.distance.cdist(y_pred[i, :2].reshape(
            [1, 2]), y_true2[:, :2], 'euclidean')
        idx = np.where(dis <= maxd)[1]
        minangle = 2*np.pi
        minidx = 0
        for j in range(idx.shape[0]):
            angle = abs(np.mod(y_pred[i, 2], 2*np.pi) - y_true2[idx[j], 2])
            angle = np.asarray([angle, 2*np.pi-angle]).min()
            if angle < minangle:
                minangle = angle
                minidx = j
        if minangle <= maxo:
            TP += 1
            y_true2 = np.delete(y_true2, idx[minidx], axis=0)

    precision = TP/float(y_pred.shape[0])
    recall = TP/float(y_true.shape[0])

    return precision, recall, y_true.shape[0], 0, 0

from scipy import spatial

def metric_P_R_F(y_true, y_pred, maxd=16, maxo=np.pi/6):
    # Calculate Precision, Recall, F-score
    if y_pred.shape[0]==0 or y_true.shape[0]==0:
        return 0,0

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    total_gt, total = float(y_true.shape[0]), float(y_pred.shape[0])
    # Using L2 loss
    dis = spatial.distance.cdist(y_pred[:, :2], y_true[:, :2], 'euclidean')
    mindis,idx = dis.min(axis=1),dis.argmin(axis=1)

    #Change to adapt to new annotation: old version. When training, comment it
    # y_pred[:,2] = -y_pred[:,2]

    angle = abs(np.mod(y_pred[:,2],2*np.pi) - y_true[idx,2])
    angle = np.asarray([angle, 2*np.pi-angle]).min(axis=0)

    # Satisfy the threshold
    # tmp=(mindis <= maxd) & (angle<=maxo)
    #print('mindis,idx,angle,tmp=%s,%s,%s,%s'%(mindis,idx,angle,tmp))

    true_match = np.unique(idx[(mindis <= maxd) & (angle<=maxo)])
    precision = len(true_match)/float(y_pred.shape[0])
    recall = len(true_match)/float(y_true.shape[0])
    # #print('pre=%f/ %f'%(len(np.unique(idx[(mindis <= maxd) & (angle<=maxo)])),float(y_pred.shape[0])))
    # #print('recall=%f/ %f'%(len(np.unique(idx[(mindis <= maxd) & (angle<=maxo)])),float(y_true.shape[0])))
    # if recall!=0:
    #     loc = np.mean(mindis[(mindis <= maxd) & (angle<=maxo)])
    #     ori = np.mean(angle[(mindis <= maxd) & (angle<=maxo)])
    # else:
    #     loc = 0
    #     ori = 0
    return precision, recall


def get_all_images(base_path, level):
    print(f'start loading data in {base_path}')
    pwd = glob(f'{base_path}/*')
    for i in range(level-1):
        father = pwd
        son = []
        for j in father:
            son += glob(f'{j}/*')
        pwd = son
    return pwd


def mnt_reader_txt(file_name):
	f = open(file_name)
	ground_truth = []
	for i, line in enumerate(f):
		if i < 2 or len(line) == 0: continue
		try:
			w, h, o = [float(x) for x in line.split()]
			w, h = int(round(w)), int(round(h))
			ground_truth.append([w, h, o])
		except:
			try:
				w, h, o, _ = [float(x) for x in line.split()]
				w, h = int(round(w)), int(round(h))
				ground_truth.append([w, h, o])
			except:
				try:
					w, h, o, _, _ = [float(x) for x in line.split()]
					w, h = int(round(w)), int(round(h))
					ground_truth.append([w, h, o])
				except:
					pass
	f.close()
	return ground_truth

def convet_gray2RGB(img):
    import cv2 as cv
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    img = img.astype(np.uint8)
    if img.ndim == 3: return img
    return cv.cvtColor(img, cv.COLOR_GRAY2BGR)


def draw_ori_on_a_image(img, ori, seg, step=8, color=(255, 255, 255)):
    import cv2 as cv
    img = img.copy()
    if img.ndim == 2: img = convet_gray2RGB(img)
    w, h, _ = img.shape
    for i, ii in enumerate(range(0, w, step)):
        for j, jj in enumerate(range(0, h, step)):
            if seg[i, j] > 0.5:
                s2t, c2t = ori[:, i, j]
                theta = np.arctan2(s2t, c2t) / 2
                cv.line(img, (jj, ii), (int(jj + step *np.cos(theta)), int(ii + step * np.sin(theta))), color)
    return img


def gaussian_kernel(shape=(5, 5), sigma=0.5):
    m, n = shape
    x = torch.arange(-(m-1) // 2, (m+1)// 2).view(-1, 1)
    y = torch.arange(-(n-1) // 2, (n+1)// 2)
    h = torch.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    kernel =  h / h.sum()
    return kernel

# Gkernel = gaussian_kernel((25, 25), sigma= 0.3*((25-1)*0.5 - 1) + 0.8).repeat((2, 1,1,1))
# sobel_kernelx = torch.tensor([[2, 2, 4, 2, 2],[1, 1, 2, 1, 1], [0, 0, 0, 0, 0], [-1, -1, -2, -1, -1], [-2, -2, -4, -2, -2]], dtype=torch.float32)
# sobel_kernely = sobel_kernelx.T
# sobel_kernelx = sobel_kernelx.reshape(1, 1, 5, 5)
# sobel_kernely = sobel_kernely.reshape(1, 1, 5, 5)
def apply_sobel(image):
    if isinstance(image, np.ndarray): image = torch.from_numpy(image)
    if image.ndim < 4: image = image.reshape(-1, 1, image.size(-2), image.size(-1))
    fx = torch.nn.functional.conv2d(image, sobel_kernelx, bias=None, stride=1, padding=2, dilation=1, groups=1)
    fy = torch.nn.functional.conv2d(image, sobel_kernely, bias=None, stride=1, padding=2, dilation=1, groups=1)
    theta = torch.atan2(fx, fy) + np.pi / 2
    ori = torch.cat((torch.sin(2 * theta), torch.cos(2 * theta)), dim=1)
    ori = torch.nn.functional.conv2d(ori, Gkernel, bias=None, stride=1, padding=12, dilation=1, groups=2)
    ori = torch.nn.functional.interpolate(ori, scale_factor= 1 / 8, recompute_scale_factor=True, mode ='bilinear', align_corners=True)
    return ori.numpy()
    #image = draw_ori_on_a_image(image, ori.squeeze().numpy(), seg)


def draw_g_on_a_image(img, seg, color=(255, 255, 255)):
    import cv2 as cv
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    if img.ndim == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ori = apply_sobel(img.astype(np.float32) / 255)
    ori = np.squeeze(ori)
    img = draw_ori_on_a_image(img, ori, seg, color=color)
    return img


def oriReader(ori):
    with open(ori, 'rb') as f:
        c = int.from_bytes(f.read(4), 'little')
        w = int.from_bytes(f.read(4), 'little')
        h = int.from_bytes(f.read(4), 'little')
        ori = np.frombuffer(f.read(-1), np.float32)
    return ori.reshape(c, w, h).copy()


# def 

def shift(img, enhance, seg, ori, offset, factor=8):
    img = ndimage.shift(img, offset, mode='constant',cval=img.mean())
    enhance = ndimage.shift(enhance, offset, mode='constant',cval=img.mean())
    offset /= factor
    seg = ndimage.shift(seg, offset, mode='constant')
    sin2ori = ndimage.shift(ori[0], offset, mode='constant') 
    cos2ori = ndimage.shift(ori[1], offset, mode='constant') 
    return img, enhance, seg, np.stack((sin2ori, cos2ori), 0)


def rotation(img, enhance, seg, ori, deagree):
    img = ndimage.rotate(img, deagree, reshape=False, mode='constant',cval=img.mean())
    enhance = ndimage.rotate(enhance, deagree, reshape=False, mode='constant',cval=img.mean())
    seg = ndimage.rotate(seg, deagree, reshape=False, mode='constant')

    sin2ori = ndimage.rotate(ori[0], deagree, reshape=False, mode='constant')
    cos2ori = ndimage.rotate(ori[1], deagree, reshape=False, mode='constant')
    deagree = (deagree * 2) / 180 * np.pi
    ori = np.arctan2(sin2ori,cos2ori) - deagree

    return img, enhance, seg, np.stack((np.sin(ori), np.cos(ori)), 0)


def shape_to_expected_shape(image, shape):
    tw, th = image.shape
    lw, lh = shape

    if tw > lw:
        s = (tw - lw) // 2
        image = image[s:s+lw, :]
    else:
        image = np.pad(image, ((0, lw-tw), (0, 0)), constant_values=0)

    if th> lh:
        s = (th - lh) // 2
        image = image[:, s:s+lh]
    else:
        image = np.pad(image, ((0, 0), (0, lh-th)), constant_values=0)

    return image


def get_all_files(p, exts=['bmp', 'png', 'jpg']):
    res = []
    files = glob(os.path.join(p, '*'))
    print(os.path.join(p, '*'), len(files))
    for f in files:
        if os.path.isfile(f):
            ext = f.split('.')[-1]
            if ext in exts:
                res.append(f)
        elif os.path.isdir(f):
            res += get_all_files(f, exts)
    return res


def vecReader(p):
    with open(p, 'rb') as f:
        dim = int.from_bytes(f.read(4), 'little')
        shape = [int.from_bytes(f.read(4), 'little') for i in range(dim)]
        ori = np.frombuffer(f.read(-1), np.float32).reshape(*shape).copy()
    return ori


def oriReader(p):
    with open(p, 'rb') as f:
        c = int.from_bytes(f.read(4), 'little')
        w = int.from_bytes(f.read(4), 'little')
        h = int.from_bytes(f.read(4), 'little')
        ori = np.frombuffer(f.read(-1), np.float32)
    return ori.reshape(c, w, h).copy()
    

def BytesReader(p):
    ext = p.split('/')[-1].split('.')[-1]
    if ext == 'ori':
        return oriReader(p)
    elif ext == 'vec':
        return vecReader(p)



class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class TensorboardLogger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head='scalar', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)

    def flush(self):
        self.writer.flush()


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            if model_ema is not None:
                to_save['model_ema'] = get_state_dict(model_ema)

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        if model_ema is not None:
            client_state['model_ema'] = get_state_dict(model_ema)
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    output_dir = Path(args.output_dir)
    if loss_scaler is not None:
        # torch.amp
        if args.auto_resume and len(args.resume) == 0:
            import glob
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
            print("Auto resume checkpoint: %s" % args.resume)

        if args.resume:
            if args.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'])
            #model_without_ddp.load_state_dict(checkpoint)
            print("Resume checkpoint %s" % args.resume)
            if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                args.start_epoch = checkpoint['epoch'] + 1
                if hasattr(args, 'model_ema') and args.model_ema:
                    _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                print("With optim & sched!")
    else:
        # deepspeed, only support '--auto_resume'.
        if args.auto_resume:
            import glob
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(output_dir, 'checkpoint-%d' % latest_ckpt)
                print("Auto resume checkpoint: %d" % latest_ckpt)
                _, client_states = model.load_checkpoint(args.output_dir, tag='checkpoint-%d' % latest_ckpt)
                args.start_epoch = client_states['epoch'] + 1
                if model_ema is not None:
                    if args.model_ema:
                        _load_checkpoint_for_ema(model_ema, client_states['model_ema'])


def create_ds_config(args):
    args.deepspeed_config = os.path.join(args.output_dir, "deepspeed_config.json")
    with open(args.deepspeed_config, mode="w") as writer:
        ds_config = {
            "train_batch_size": args.batch_size * args.update_freq * get_world_size(),
            "train_micro_batch_size_per_gpu": args.batch_size,
            "steps_per_print": 1000,
            "optimizer": {
                "type": "Adam",
                "adam_w_mode": True,
                "params": {
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "bias_correction": True,
                    "betas": [
                        0.9,
                        0.999
                    ],
                    "eps": 1e-8
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 7,
                "loss_scale_window": 128
            }
        }

        writer.write(json.dumps(ds_config, indent=2))
