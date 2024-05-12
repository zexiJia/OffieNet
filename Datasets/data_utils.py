import numpy as np
from skimage.exposure import adjust_gamma,rescale_intensity

def mnt_reader(file_name):
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


def add_salt_noise(img,p = np.random.rand()/6):
	image0=img
	mval = 210
	for i in range(int(p*image0.shape[0]*image0.shape[1]/100)):
		x=np.random.randint(0,image0.shape[0])
		y=np.random.randint(0,image0.shape[1])
		image0[max(x-3,0):min(x+3,image0.shape[0]-1),max(y-3,0):min(y+3,image0.shape[1]-1)]=mval
	return image0

def add_latent_noise(image0):
	# image0=np.matrix(img)
	# print image0.mean()
	image0[image0>=min(image0.mean()+70,220)]=image0.mean()
	image0[image0<=max(image0.mean()-70,40)]=image0.mean()
	if np.random.rand()<0.8:
		image0 = adjust_gamma(image0,5)
	else:
		image0 = adjust_gamma(image0,0.5)
	image0=rescale_intensity(image0)
	return image0


def texture_fn(ksize, sigma, theta, Lambda, psi, gamma):
	sigma_x = sigma
	sigma_y = float(sigma) / gamma
	nstds = 3
	xmax = ksize[0]/2
	ymax = ksize[1]/2
	xmin = -xmax
	ymin = -ymax
	(y, x) = np.meshgrid(np.arange(ymin, ymax ), np.arange(xmin, xmax ))
	x_theta = x * np.cos(theta) + y * np.sin(theta)
	# y_theta = -x * np.sin(theta) + y * np.cos(theta)
	gb_cos = np.cos(2 * np.pi / Lambda * x_theta + psi)
	# gb_sin = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.sin(2 * np.pi / Lambda * x_theta + psi)
	return gb_cos

def add_texture_noise(img):
	image0=img/255.0
	img_shape=image0.shape
	texture = texture_fn(img_shape, 4.5, -np.pi*0.25, 8, 0, 0.5)
	## generate random mask
	mask = np.zeros(img_shape)
	# kk = 200
	# xxx = np.random.randint(kk+1,image0.shape[0]-kk-1)
	# yyy = np.random.randint(kk+1,image0.shape[1]-kk-1)
	# mask[xxx-kk:xxx+kk,yyy-kk:yyy+kk] = texture[xxx-kk:xxx+kk,yyy-kk:yyy+kk]
	for ii in range(img_shape[1]):
		mask[:ii+100,ii] = texture[:ii+100,ii]
	mask[mask>-0.8]=0
	image0 = image0+mask*0.6
	image0[image0>1]=1
	image0[image0<0]=0
	image0=image0*255
	return image0