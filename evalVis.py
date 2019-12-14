import json
import os.path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from skimage import transform, filters
from tqdm import tqdm

import misc.config as config
import misc.dataEval as data
import misc.modelEval as model
import misc.utils as utils
from misc.aleatoric_loss import AleatoricCrossEntropyLoss


def get_p_gradcam(grads_val, target):
   cams = []
   for i in range(grads_val.shape[0]):
      weights = np.mean(grads_val[i], axis=(1, 2))
      cam = np.zeros(target[i].shape[1:], dtype=np.float32)

      for k, w in enumerate(weights):
         cam += w * target[i, k, :, :]

      cams.append(cam)

   return cams


def get_blend_map_gradcam(img, gradcam_map):
   cam = np.maximum(gradcam_map, 0)
   cam = cv2.resize(cam, img.shape[:2])
   cam = cam - np.min(cam)
   cam = cam / np.max(cam)
   heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
   heatmap = np.float32(heatmap) / 255
   cam = heatmap + np.float32(img)
   cam = cam / np.max(cam)
   return cam


def get_blend_map_att(img, att_map, blur=True, overlap=True):
   att_map -= att_map.min()
   if att_map.max() > 0:
      att_map /= att_map.max()
   att_map = att_map.reshape((14, 14))
   att_map = transform.resize(att_map, (img.shape[:2]), order=3)
   if blur:
      att_map = filters.gaussian(att_map, 0.01 * max(img.shape[:2]))
      att_map -= att_map.min()
      att_map /= att_map.max()
   cmap = plt.get_cmap('jet')
   att_map_v = cmap(att_map)
   att_map_v = np.delete(att_map_v, 3, 2)
   if overlap:
      att_map = (1 - att_map ** 0.4).reshape(att_map.shape + (1,)) * img + (att_map ** 0.4).reshape(
         att_map.shape + (1,)) * att_map_v
   return att_map


def update_learning_rate(optimizer, iteration):
   lr = config.initial_lr * 0.5 ** (float(iteration) / config.lr_halflife)
   for param_group in optimizer.param_groups:
      param_group['lr'] = lr


total_iterations = 0


def run(net, net_var, loader, optimizer, optimizer_var, tracker, train=False, prefix='', epoch=0):
   count = 0
   with open('inputImages.json',
             encoding="utf8") as f:  # inputImages.json: keys are the questions for which we require visualisation
      diff = json.load(f)

   folderPre = ""
   desired = []
   for key, value in diff.items():
      desired.append(int(key))

   desired = sorted(desired)

   net.train()
   net_var.train()
   tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
   answ = []
   idxs = []
   accs = []

   tq = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0)

   log_softmax = nn.LogSoftmax().cuda()
   aleatoric_loss = AleatoricCrossEntropyLoss().cuda()
   for v, q, a, idx, image_id, q_len in tq:

      torch.cuda.empty_cache()
      var_params = {
         'volatile': train,
         'requires_grad': True,
      }

      forward = False
      imgList = [int(img) for img in image_id]
      currIms = []

      for im in imgList:
         if im in desired:
            currIms.append(im)
            forward = True
      if not forward:
         continue

      v = Variable(v.cuda(async = True), ** var_params)
      q = Variable(q.type(torch.FloatTensor).cuda(async = True), ** var_params)
      a = Variable(a.type(torch.FloatTensor).cuda(async = True), ** var_params)
      q_len = Variable(q_len.type(torch.FloatTensor).cuda(async = True), ** var_params)

      out, p_at = net(v, q, q_len)
      nll = -log_softmax(out)

      ud_ce_loss = (nll * a / 10).sum(dim=1).mean()

      logits_variance = net_var(out)
      gce_loss, variance_loss, undistorted_loss, variance_depressor = aleatoric_loss(logits_variance, out, a)
      aleatoric_uncertainty_loss = gce_loss + variance_loss + variance_depressor
      loss = ud_ce_loss + aleatoric_uncertainty_loss

      loss.backward(retain_graph=True)
      aleatoric_uncertainty_loss.backward(retain_graph=True)
      gradients = get_p_gradcam(v.grad.cpu().data.numpy(), v.cpu().data.numpy())
      for i, imgIdx in enumerate(image_id):
         if int(imgIdx) not in currIms:
            continue
         count += 1
         if count == 1001:
            quit()
         qd = int(imgIdx)
         imgIdx = imgIdx // 10
         imgIdx = "COCO_" + prefix + "2014_000000" + "0" * (6 - len(str(imgIdx.numpy()))) + str(
            imgIdx.numpy()) + ".jpg"
         rawImg = scipy.misc.imread(os.path.join(
            'VQA/Images/mscoco/',  # change the directory to VQA mscoco of your system
            prefix + '2014/' + imgIdx), mode='RGB')
         rawImg = scipy.misc.imresize(rawImg, (448, 448), interp='bicubic')
         plt.imsave("Results" + folderPre + "/RawImages/" + str(qd) + ".png", rawImg)

         plt.imsave("Results" + folderPre + "/AttImages/" + str(qd) + ".png",
                    get_blend_map_att(rawImg / 255.0, p_at[i].cpu().data.numpy()))
         cv2.imwrite("Results" + folderPre + "/GradcamImages/" + str(qd) + ".png",
                     np.uint8(255 * get_blend_map_gradcam(rawImg / 255.0, gradients[i])))


def main():
   if len(sys.argv) > 1:
      name = ' '.join(sys.argv[1:])
   else:
      from datetime import datetime
      name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
   target_name = os.path.join('logs', '{}.pth'.format(name))
   print('will save to {}'.format(target_name))

   cudnn.benchmark = True

   train_loader = data.get_loader(train=True)
   val_loader = data.get_loader(val=True)

   net = nn.DataParallel(model.Net(train_loader.dataset.num_tokens)).cuda()
   optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad])

   net_var = nn.DataParallel(model.Uncertainty(config.max_answers)).cuda()
   optimizer_var = optim.SGD([p for p in net_var.parameters() if p.requires_grad], lr=0.0002)

   tracker = utils.Tracker()

   ckp = torch.load('logs/2019-03-19_22:49:23.pth_9.pth')
   net.load_state_dict(ckp['weights'])
   net_var.load_state_dict(ckp['weights_var'])

   run(net, net_var, val_loader, optimizer, optimizer_var, tracker, train=False, prefix='val')


if __name__ == '__main__':
   main()
