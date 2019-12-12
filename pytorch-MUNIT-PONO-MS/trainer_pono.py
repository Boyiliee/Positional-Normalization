"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from networks_pono import (AdaINGen, MsImageDis, VAEGen, ClassEncoder, VGG19,
    AdaINGen_PONO, SpatialNorm)
from utils import weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class MUNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(MUNIT_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # if 'arch' not in hyperparameters['gen']:
        #     hyperparameters['gen']['arch'] = 'adain'
        # if 'arch' not in hyperparameters['dis']:
        #     hyperparameters['dis']['arch'] = 'msd'
        
        # Initiate the networks
        print(hyperparameters['gen'])
        if hyperparameters['gen']['arch'] == 'adain_ponoms':
            self.gen_a = AdaINGen_PONO(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
            self.gen_b = AdaINGen_PONO(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        else:
            self.gen_a = AdaINGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
            self.gen_b = AdaINGen(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b

        self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b

        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters['gen']['style_dim']
        if hyperparameters['cls_w'] > 0:
            self.cls = ClassEncoder(4, self.gen_a.enc_content.output_dim, hyperparameters['cls']['max_dim'], 8, norm='none',
                                    activ=hyperparameters['gen']['activ'], pad_type=hyperparameters['gen']['pad_type'])
        else:
            self.cls = None

        # print(self.gen_a)

        # fix the noise used in sampling
        display_size = int(hyperparameters['display_size'])
        self.s_a = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.s_b = torch.randn(display_size, self.style_dim, 1, 1).cuda()

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        if self.cls:
            gen_params += list(self.cls.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))

        # Load VGG model if needed
        if 'vgg_w' in hyperparameters and 'vgg_w_a' not in hyperparameters:
            hyperparameters['vgg_w_a'] = hyperparameters['vgg_w']
        if 'vgg_w' in hyperparameters and 'vgg_w_b' not in hyperparameters:
            hyperparameters['vgg_w_b'] = hyperparameters['vgg_w']

        if 'vgg_w_a' in hyperparameters.keys() and hyperparameters['vgg_w_a'] > 0 or 'vgg_w_b' in hyperparameters and hyperparameters['vgg_w_b'] > 0:
            # self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg = VGG19(init_weights=hyperparameters['vgg_model_path'], feature_mode=True)
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
        print('# Dis params: {}'.format(sum([p.numel() for p in dis_params])))
        print('# Gen params: {}'.format(sum([p.numel() for p in gen_params])))

    def recon_criterion(self, input, target):
        if torch.is_tensor(input):
            return torch.mean(torch.abs(input - target))
        elif torch.is_tensor(input[1]):
            loss = F.l1_loss(input[0], target[0])
            for m1, m2 in zip(input[1], target[1]):
                loss += F.l1_loss(m1, m2)
            return loss
        else:
            loss = F.l1_loss(input[0], target[0])
            for (m1, s1), (m2, s2) in zip(input[1], target[1]):
                loss += F.l1_loss(m1, m2) + F.l1_loss(s1, s2)
            return loss

    def forward(self, x_a, x_b):
        self.eval()
        s_a = Variable(self.s_a)
        s_b = Variable(self.s_b)
        c_a, s_a_fake = self.gen_a.encode(x_a)
        c_b, s_b_fake = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        self.train()
        return x_ab, x_ba

    def gen_update(self, x_a, x_b, hyperparameters):
        self.gen_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_a, s_a_prime = self.gen_a.encode(x_a)
        c_b, s_b_prime = self.gen_b.encode(x_b)
        if self.decode_with_images:
            # decode (within domain)
            x_a_recon = self.gen_a.decode(c_a, s_a_prime, x_a)
            x_b_recon = self.gen_b.decode(c_b, s_b_prime, x_b)
            # decode (cross domain)
            x_ba = self.gen_a.decode(c_b, s_a, x_b)
            x_ab = self.gen_b.decode(c_a, s_b, x_a)
        else:
            # decode (within domain)
            x_a_recon = self.gen_a.decode(c_a, s_a_prime)
            x_b_recon = self.gen_b.decode(c_b, s_b_prime)
            # decode (cross domain)
            x_ba = self.gen_a.decode(c_b, s_a)
            x_ab = self.gen_b.decode(c_a, s_b)
        # encode again
        c_b_recon, s_a_recon = self.gen_a.encode(x_ba)
        c_a_recon, s_b_recon = self.gen_b.encode(x_ab)
        # decode again (if needed)
        if self.decode_with_images:
            x_aba = self.gen_a.decode(c_a_recon, s_a_prime, x_ab) if hyperparameters['recon_x_cyc_w'] > 0 else None
            x_bab = self.gen_b.decode(c_b_recon, s_b_prime, x_ba) if hyperparameters['recon_x_cyc_w'] > 0 else None
        else:
            x_aba = self.gen_a.decode(c_a_recon, s_a_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None
            x_bab = self.gen_b.decode(c_b_recon, s_b_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a)
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b)
        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
        self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b)
        self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
        # domain-invariant perceptual loss
        # self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
        # self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_a = self.compute_vgg19_loss(self.vgg, x_ab, x_a, vgg_type=hyperparameters['vgg_type']) if hyperparameters['vgg_w_a'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg19_loss(self.vgg, x_ba, x_b, vgg_type=hyperparameters['vgg_type']) if hyperparameters['vgg_w_b'] > 0 else 0
        # print("VGG:", self.loss_gen_adv_a.item(), self.loss_gen_adv_b.item())
        # class loss
        # if hyperparameters['cls_w'] > 0:
        #     c_a_pred = self.cls(c_a)
        #     c_a_recon_pred = self.cls(c_a_recon)
        #     self.loss_gen_cls_c_a = F.cross_entropy(c_a_pred, y_a)
        #     self.loss_gen_cls_c_a_recon = F.cross_entropy(c_a_recon_pred, y_a)
        #     with torch.no_grad():
        #         self.loss_gen_cls_c_a_err = c_a_pred.argmax(dim=1).ne(y_a).float().mean()
        #         self.loss_gen_cls_c_a_recon_err = c_a_recon_pred.argmax(dim=1).ne(y_a).float().mean()
        #     print(f'cls_a_err: {self.loss_gen_cls_c_a_err.item():.3f} cls_a_recon_err: {self.loss_gen_cls_c_a_recon_err.item():.3f}')
        # else:
        #     self.loss_gen_cls_c_a = 0
        #     self.loss_gen_cls_c_a_recon = 0
        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b + \
                              hyperparameters['vgg_w_a'] * self.loss_gen_vgg_a + \
                              hyperparameters['vgg_w_b'] * self.loss_gen_vgg_b
        self.loss_gen_total.backward()
        self.gen_opt.step()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def compute_vgg19_loss(self, vgg, img, target, vgg_type='vgg19'):
        img_feature = self.vgg((img + 1) / 2)
        target_feature = self.vgg((target + 1) / 2).detach()
        if vgg_type == 'vgg19':
            return F.l1_loss(img_feature, target_feature)
        if vgg_type == 'vgg19_sp':
            sp = SpatialNorm(affine=False)
            return F.l1_loss(sp(img_feature)[0], sp(target_feature)[0])
        elif vgg_type == 'vgg19_sp_mean':
            m1, m2 = img_feature.mean(dim=1), target_feature.mean(dim=1)
            return F.l1_loss(m1, m2)
        elif vgg_type == 'vgg19_sp_mean_mix':
            m1, m2 = img_feature.mean(dim=1), target_feature.mean(dim=1)
            return 0.5 * F.l1_loss(img_feature, target_feature) + 0.5 * F.l1_loss(m1, m2)
        elif vgg_type == 'vgg19_sp_meanstd':
            m1, m2 = img_feature.mean(dim=1), target_feature.mean(dim=1)
            std1, std2 = img_feature.std(dim=1), target_feature.std(dim=1)
            return 0.5 * F.l1_loss(m1, m2) + 0.5 *F.l1_loss(std1, std2)
        elif vgg_type == 'vgg19_sp_meanstd_mix':
            m1, m2 = img_feature.mean(dim=1), target_feature.mean(dim=1)
            std1, std2 = img_feature.std(dim=1), target_feature.std(dim=1)
            return 0.5 * F.l1_loss(img_feature, target_feature) + 0.25 * F.l1_loss(m1, m2) + 0.25 * F.l1_loss(std1, std2)
        elif vgg_type == 'vgg19_in':
            return F.l1_loss(F.instance_norm(img_feature), F.instance_norm(target_feature))
        elif vgg_type == 'vgg19_in_mean':
            img_feature = img_feature.view(*img_feature.shape[:2], -1)
            target_feature = target_feature.view(*target_feature.shape[:2], -1)
            m1, m2 = img_feature.mean(dim=2), target_feature.mean(dim=2)
            return F.l1_loss(m1, m2)
        elif vgg_type == 'vgg19_in_meanstd':
            img_feature = img_feature.view(*img_feature.shape[:2], -1)
            target_feature = target_feature.view(*target_feature.shape[:2], -1)
            m1, m2 = img_feature.mean(dim=2), target_feature.mean(dim=2)
            std1, std2 = img_feature.std(dim=2), target_feature.std(dim=2)
            return F.l1_loss(m1, m2) + F.l1_loss(std1, std2)
        else:
            raise ValueError('vgg_type = {}'.format(vgg_type))

    def sample_old(self, x_a, x_b):
        self.eval()
        s_a1 = Variable(self.s_a)
        s_b1 = Variable(self.s_b)
        s_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], []
        for i in range(x_a.size(0)):
            c_a, s_a_fake = self.gen_a.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.gen_a.decode(c_a, s_a_fake))
            x_b_recon.append(self.gen_b.decode(c_b, s_b_fake))
            x_ba1.append(self.gen_a.decode(c_b, s_a1[i].unsqueeze(0)))
            x_ba2.append(self.gen_a.decode(c_b, s_a2[i].unsqueeze(0)))
            x_ab1.append(self.gen_b.decode(c_a, s_b1[i].unsqueeze(0)))
            x_ab2.append(self.gen_b.decode(c_a, s_b2[i].unsqueeze(0)))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
        x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
        self.train()
        return x_a, x_a_recon, x_ab1, x_ab2, x_b, x_b_recon, x_ba1, x_ba2

    def sample(self, x_a, x_b):
        self.eval()
        self.eval()
        s_a1 = Variable(self.s_a)
        s_b1 = Variable(self.s_b)
        x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], []
        for i in range(x_a.size(0)):
            c_a, s_a_fake = self.gen_a.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake = self.gen_b.encode(x_b[i].unsqueeze(0))
            if self.decode_with_images:
                x_a_recon.append(self.gen_a.decode(c_a, s_a_fake, x_a[i:i+1]))
                x_b_recon.append(self.gen_b.decode(c_b, s_b_fake, x_b[i:i+1]))
                x_ba1.append(self.gen_a.decode(c_b, s_a1[i].unsqueeze(0), x_b[i:i+1]))
                x_ba2.append(self.gen_a.decode(c_b, s_a_fake, x_b[i:i+1]))
                x_ab1.append(self.gen_b.decode(c_a, s_b1[i].unsqueeze(0), x_a[i:i+1]))
                x_ab2.append(self.gen_b.decode(c_a, s_b_fake, x_a[i:i+1]))
            else:
                x_a_recon.append(self.gen_a.decode(c_a, s_a_fake))
                x_b_recon.append(self.gen_b.decode(c_b, s_b_fake))
                x_ba1.append(self.gen_a.decode(c_b, s_a1[i].unsqueeze(0)))
                x_ba2.append(self.gen_a.decode(c_b, s_a_fake))
                x_ab1.append(self.gen_b.decode(c_a, s_b1[i].unsqueeze(0)))
                x_ab2.append(self.gen_b.decode(c_a, s_b_fake))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
        x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
        self.train()
        return x_a, x_a_recon, x_ab1, x_ab2, x_b, x_b, x_b_recon, x_ba1, x_ba2, x_a

    def dis_update(self, x_a, x_b, hyperparameters):
        self.dis_opt.zero_grad()
        dis_w_true = hyperparameters['gan_dis_w_true'] if 'gan_dis_w_true' in hyperparameters else 0
        dis_w_rand = hyperparameters['gan_dis_w_rand'] if 'gan_dis_w_rand' in hyperparameters else hyperparameters['gan_w']
        if dis_w_rand > 0.:
            s_a_rand = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
            s_b_rand = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # encode
        if dis_w_rand > 0. or dis_w_true > 0.:
            c_a, s_a = self.gen_a.encode(x_a)
            c_b, s_b = self.gen_b.encode(x_b)
        # decode (cross domain)
        if dis_w_true > 0.:
            if self.decode_with_images:
                x_ba = self.gen_a.decode(c_b, s_a, x_b)
                x_ab = self.gen_b.decode(c_a, s_b, x_a)
            else:
                x_ba = self.gen_a.decode(c_b, s_a)
                x_ab = self.gen_b.decode(c_a, s_b)
            self.loss_dis_a_true = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
            self.loss_dis_b_true = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
            self.loss_dis_true = self.loss_dis_a_true + self.loss_dis_b_true
            self.loss_dis_true.backward(retain_graph=True)
        else:
            self.loss_dis_true = 0.

        if dis_w_rand > 0.:
            if self.decode_with_images:
                x_ba_rand = self.gen_a.decode(c_b, s_a_rand, x_b)
                x_ab_rand = self.gen_b.decode(c_a, s_b_rand, x_a)
            else:
                x_ba_rand = self.gen_a.decode(c_b, s_a_rand)
                x_ab_rand = self.gen_b.decode(c_a, s_b_rand)
            self.loss_dis_a_rand = self.dis_a.calc_dis_loss(x_ba_rand.detach(), x_a)
            self.loss_dis_b_rand = self.dis_b.calc_dis_loss(x_ab_rand.detach(), x_b)
            self.loss_dis_rand = self.loss_dis_a_rand + self.loss_dis_b_rand
            self.loss_dis_rand.backward(retain_graph=True)
        else:
            self.loss_dis_rand = 0.

        # D loss
        self.loss_dis_total = dis_w_true * self.loss_dis_true + dis_w_rand * self.loss_dis_rand
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        if self.cls is not None:
            self.cls.load_state_dict(state_dict['cls'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict(),
                    'cls': self.cls.state_dict() if self.cls is not None else None}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)


