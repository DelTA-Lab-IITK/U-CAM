import math
import os.path
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import misc.config as config
import misc.dataEval as data
import misc.modelEval as model
import misc.utils as utils


def run(net, net_var, loader, optimizer, optimizer_var, tracker, train=False, prefix='', epoch=0):
   count = 0
   COUNT = 20000  # calculating classification error and entropy of 20,000 randomly sampled questions
   N_MC = 50  # no. of Monte-carlo simulations
   cps = []
   cvs = []
   wps = []
   wvs = []

   net.train()
   net_var.train()

   tq = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0)
   for v, q, a, idx, image_id, q_len in tq:
      p_miss = []
      if count > COUNT:
         break

      var_params = {
         'volatile': train,
         'requires_grad': True,
      }

      v = Variable(v.cuda(async = True), ** var_params)
      q = Variable(q.type(torch.FloatTensor).cuda(async = True), ** var_params)
      a = Variable(a.type(torch.FloatTensor).cuda(async = True), ** var_params)
      q_len = Variable(q_len.type(torch.FloatTensor).cuda(async = True), ** var_params)

      a_temp = a
      a_temp = a_temp.detach().cpu().numpy()
      a_indices = np.argmax(a_temp, axis=1)

      out, p_at = net(v, q, q_len)
      sum = np.zeros(tuple(out.shape))

      for j in range(N_MC):
         out, p_at = net(v, q, q_len)
         preds = F.softmax(out, dim=1)
         sum += preds.data

      avg = sum / N_MC
      entropy = -1 * np.sum(avg * np.log(avg), axis=-1)

      for k, an_index in enumerate(a_indices):
         p_miss.append(1 - avg[k][an_index])  # probability of mis-classification

      acc = utils.batch_accuracy(out.data, a.data).cpu()

      for i, imgIdx in enumerate(image_id):
         if math.isnan(entropy[i]):
            continue
         if count > COUNT:
            break
         count += 1
         p = p_miss[i]
         error = np.log(1 / (1 - p))
         if acc[i] == 0:
            wps.append(error)
            wvs.append(entropy[i])
         else:
            cps.append(error)
            cvs.append(entropy[i])

   with open("classification_error_of_correct_samples.txt", "w") as file:
      file.write(str(cps))
   with open("entropy_of_correct_samples.txt", "w") as file:
      file.write(str(cvs))
   with open("classification_error_of_incorrect_samples.txt.txt", "w") as file:
      file.write(str(wps))
   with open("entropy_of_incorrect_samples.txt", "w") as file:
      file.write(str(wvs))


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
