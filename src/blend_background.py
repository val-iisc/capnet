import numpy as np
import cv2
import random
import scipy.misc as misc
import matplotlib.pyplot as plt
import os, os.path as osp

def computeBbox(fgMask, paddingFrac=0.05):
	'''
	Args:
		fgMask: input fgMask of shape (H,W)
				0--> object is absent
				1--> object is present
		paddingFrac: how much to pad on both sides
	Returns:
		Hmin, Hmax, Wmin, Wmax: Corners of the bounding box
	'''

	H = fgMask.shape[0]
	W = fgMask.shape[1]

	Hmin = np.argmin(np.max(fgMask, axis=1) * np.arange(H) + (np.max(fgMask, axis=1) == 0)*H)
	Hmax = np.argmax(np.max(fgMask, axis=1) * np.arange(H))

	Wmin = np.argmin(np.max(fgMask, axis=0) * np.arange(W) + (np.max(fgMask, axis=0) == 0)*W)
	Wmax = np.argmax(np.max(fgMask, axis=0) * np.arange(W))

	H_box = Hmax-Hmin+1
	W_box = Wmax-Wmin+1

	Hmin = max(0, int(Hmin-paddingFrac*H_box))
	Hmax = Hmax+int(paddingFrac*H_box)

	Wmin = max(0, int(Wmin-paddingFrac*W_box))
	Wmax = Wmax+int(paddingFrac*W_box)

	return (Hmin, Hmax, Wmin, Wmax)

def blendBg(fgImgPath, bgImgsList, OUT_H, OUT_W):
	'''
	Args:
		fgImgPath: path to input image of shape (H,W,4)
		bgImgsList: list of absolute paths for background images from SUN dataset
		H,W : Height, Width of the output image

	Returns:
		outImg: output image of shape (OUT_H,OUT_W,3) with random real background

	Description:
		Take a random crop from a random background, overlay this random crop using 
		alpha mask, finding bounding box for chair using fgMask, pad it to 5% on
		both sides, resize to (OUT_H,OUT_W), convert to RGB
	'''

	fgImg = cv2.imread(fgImgPath, cv2.IMREAD_UNCHANGED)
	fgMask = fgImg[:,:,3] / 255

	bgId = bgImgsList[random.randint(0, len(bgImgsList)-1)]
	bgImg = cv2.imread(bgId)

	maxHW_fg = max(fgMask.shape[0], fgMask.shape[1])
	minHW_bg = min(bgImg.shape[0] , bgImg.shape[1])

	if minHW_bg < maxHW_fg:
		rsz_ratio = float(maxHW_fg) / float(minHW_bg)
		newH_bg = int(np.ceil(bgImg.shape[0]*rsz_ratio))
		newW_bg = int(np.ceil(bgImg.shape[1]*rsz_ratio))
		bgImg = cv2.resize(bgImg, (newH_bg, newW_bg), interpolation=cv2.INTER_LINEAR)

	initH = random.randint(0, bgImg.shape[0] - fgImg.shape[0])
	initW = random.randint(0, bgImg.shape[1] - fgImg.shape[1])

	crop_bg = bgImg[initH:initH+fgMask.shape[0], initW:initW+fgMask.shape[1]]

	alpha_mask = np.expand_dims(fgMask, axis=-1)
	alpha_mask = np.tile(alpha_mask, [1,1,3])
	outImg = fgImg[:,:,:3]*alpha_mask + crop_bg*(1-alpha_mask)

	Hmin, Hmax, Wmin, Wmax = computeBbox(fgMask)

	outImg = outImg[Hmin:Hmax, Wmin:Wmax]
	outImg = cv2.resize(outImg, (OUT_H,OUT_W), interpolation=cv2.INTER_LINEAR)
	outImg = cv2.cvtColor(outImg, cv2.COLOR_BGR2RGB)

	return outImg

if __name__ == '__main__':

	sun_dir = '/data2/priyanka/Remotes/124/home/ram/priyanka/3DR/others/RenderForCNN-master/datasets/sun2012pascalformat/JPEGImages'
	bgImgsList = os.listdir(sun_dir)
	bgImgsList = [osp.join(sun_dir, img_path) for img_path in bgImgsList]

	img_path = '/data2/priyanka/Remotes/124/home/ram/priyanka/3DR/3DRModels/ShapeNet_drc/03001627/fffda9f09223a21118ff2740a556cc3/render_0.png'
	# img_path = '/data2/priyanka/Remotes/124/home/ram/priyanka/3DR/3DRModels/ShapeNet_drc/03001627/7f271ecbdeb7610d637adadafee6f182/render_0.png'
	realImgRgb = blendBg(img_path, bgImgsList, 64, 64)

	plt.imshow(realImgRgb)
	plt.show()