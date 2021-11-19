from data_loader import Dataset_JSON
from torch.utils import data
import torch
from torch.optim import Adam
from torch.autograd import grad as torch_grad
import multiprocessing
from pathlib import Path
from discriminator import Discriminator
from generator import Generator, MappingNetwork, StyleVectorizer
from tqdm import tqdm
import argparse
import torch.nn as nn
import torchvision
import os
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

COLORS_BIRD = {'initial':1-torch.cuda.FloatTensor([45, 169, 145]).view(1, -1, 1, 1)/255., 'eye':1-torch.cuda.FloatTensor([243, 156, 18]).view(1, -1, 1, 1)/255., 'none':1-torch.cuda.FloatTensor([149, 165, 166]).view(1, -1, 1, 1)/255., 
        'beak':1-torch.cuda.FloatTensor([211, 84, 0]).view(1, -1, 1, 1)/255., 'body':1-torch.cuda.FloatTensor([41, 128, 185]).view(1, -1, 1, 1)/255., 'details':1-torch.cuda.FloatTensor([171, 190, 191]).view(1, -1, 1, 1)/255.,
        'head':1-torch.cuda.FloatTensor([192, 57, 43]).view(1, -1, 1, 1)/255., 'legs':1-torch.cuda.FloatTensor([142, 68, 173]).view(1, -1, 1, 1)/255., 'mouth':1-torch.cuda.FloatTensor([39, 174, 96]).view(1, -1, 1, 1)/255., 
        'tail':1-torch.cuda.FloatTensor([69, 85, 101]).view(1, -1, 1, 1)/255., 'wings':1-torch.cuda.FloatTensor([127, 140, 141]).view(1, -1, 1, 1)/255.}

def cycle(iterable):
    while True:
        for i in iterable:
            yield i

class Trainer:
    def __init__(self, data_dir, results_dir, models_dir, img_size, batch_size, grad_acc_every, alpha_update_every, **kwargs):
        self.data_dir = data_dir
        self.results_dir = Path(results_dir)
        self.models_dir = Path(models_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.grad_acc_every = grad_acc_every
        self.alpha_update_every = alpha_update_every
        self.kwargs = kwargs
        self.train_step_counter = 0
        self.model_name = kwargs['model-name']

        self.part_to_id = {'initial': 0, 'eye': 1, 'head': 4, 'body': 3, 'beak': 2, 'legs': 5, 'wings': 8, 'mouth': 6, 'tail': 7}
        COLORS = COLORS_BIRD
        self.color = 1-torch.cuda.FloatTensor([0, 0, 0]).view(1, -1, 1, 1)
        self.default_color = 1-torch.cuda.FloatTensor([0, 0, 0]).view(1, -1, 1, 1)
        for key in COLORS:
            if key in self.model_name:
                self.color = COLORS[key]
                break

        for partname in self.part_to_id.keys():
            if partname in self.model_name:
                self.partid = self.part_to_id[partname]
                self.partname = partname

    def init_gen(self):
        self.map_net = MappingNetwork(in_size=self.img_size, in_ch=self.kwargs['img-channels'], latent_dim=self.kwargs['latent-dim'])
        self.style_vec = StyleVectorizer(self.kwargs['latent-dim'], self.kwargs['latent-dim'], 0.2)
        self.gen = Generator(self.img_size, self.kwargs['latent-dim'], reversed(self.map_net.channels), img_channels=1, concat_map=True)
        if torch.cuda.is_available():
            self.map_net.cuda()
            self.gen.cuda()

        gen_params = list(self.map_net.parameters()) + list(self.gen.parameters()) + list(self.style_vec.parameters())
        self.gen_opt = Adam(gen_params, lr=self.kwargs['lr-g'], betas=(0, 0.99)) # WGAN-gp uses Adam

    def init_critic(self):
        self.critic = Discriminator(self.img_size, img_channels=self.kwargs['img-channels'])
        if torch.cuda.is_available(): self.critic.cuda()
        self.critic_opt = Adam(self.critic.parameters(), lr=self.kwargs['lr-d'], betas=(0, 0.99)) # WGAN-gp uses Adam

    def set_data_src(self):
        data_dir = self.data_dir
        large_aug = self.kwargs['large-augmentation']
        num_cores = multiprocessing.cpu_count()
        self.dataset_D = Dataset_JSON(data_dir, self.img_size, large_aug=large_aug)
        self.dataset_G = Dataset_JSON(data_dir, self.img_size, large_aug=large_aug)
        self.loader_D = cycle(data.DataLoader(self.dataset_D, num_workers = int(num_cores/2), batch_size = self.batch_size, drop_last = True, shuffle=True, pin_memory=True))
        self.loader_G = cycle(data.DataLoader(self.dataset_G, num_workers = int(num_cores/2), batch_size = self.batch_size, drop_last = True, shuffle=True, pin_memory=True))

    def downsample(self, img, to_size):
        pooler = nn.AvgPool2d(kernel_size=2, stride=2)
        while img.size(-1) != to_size:
            img = pooler(img)
        return img

    def gradient_penalty(self, images, output, weight = 10):
        batch_size = images.shape[0]
        gradients = torch_grad(outputs=output, inputs=images,
                            grad_outputs=torch.ones(output.size()).cuda(),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(batch_size, -1)
        return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    def noise(self, n, latent_dim):
        return torch.randn(n, latent_dim).cuda()

    def noise_list(self, n, layers, latent_dim):
        return [(self.noise(n, latent_dim), layers)]
    
    def mixed_list(self, n, layers, latent_dim):
        tt = int(torch.rand(()).numpy() * layers)
        return self.noise_list(n, tt, latent_dim) + self.noise_list(n, layers - tt, latent_dim) 

    def train_step(self, alpha, steps, use_sep_ln, flip_p, use_gp_loss, use_label_noise, loss_to_use, factor_critic=1):
        self.critic_opt.zero_grad()

        for _ in range(factor_critic * self.grad_acc_every):
            image_batch, image_cond_batch, part_only_batch = [item.cuda() for item in next(self.loader_D)]
            image_partial_batch = image_cond_batch[:, -1:, :, :]

            latent_vector, cond_feat_maps = self.map_net(image_cond_batch)
            if use_sep_ln:
                latent_vector = None
                noise = self.noise(self.batch_size, self.kwargs['latent-dim'])
                latent_vector = self.style_vec(noise)

            gen_imgs = self.gen(latent_vector, alpha, steps, cond_feature_maps=cond_feat_maps) # steps will be updated in the wrapper function based on the returned value of alpha

            image_batch = self.downsample(image_batch, to_size=gen_imgs.size(-1))
            image_cond_batch = self.downsample(image_cond_batch, to_size=gen_imgs.size(-1))
            part_only_batch = self.downsample(part_only_batch, to_size=gen_imgs.size(-1))
            image_partial_batch = self.downsample(image_partial_batch, to_size=gen_imgs.size(-1))

            gen_imgs_add = torch.max(gen_imgs, image_partial_batch)

            generated_image_stack_batch = torch.cat([image_cond_batch[:, :self.partid], torch.max(gen_imgs, image_cond_batch[:, self.partid:self.partid+1]),
                                                    image_cond_batch[:, self.partid+1:-1], gen_imgs_add], 1)

            fake_proba = self.critic(generated_image_stack_batch.clone().detach(), alpha, steps)
            real_image_stack_batch = torch.cat([image_cond_batch[:, :self.partid], torch.max(part_only_batch, image_cond_batch[:, self.partid:self.partid+1]),
                                                        image_cond_batch[:, self.partid+1:-1], image_batch], 1)

            fooling_noise = torch.randn(real_image_stack_batch.size()).to(device)/(0.05 * self.train_step_counter + 1)
            real_image_stack_batch = real_image_stack_batch + fooling_noise

            real_image_stack_batch.requires_grad_()
            real_proba = self.critic(real_image_stack_batch, alpha, steps)
            
            # Introducing label noise
            if torch.rand(1) <= flip_p and use_label_noise:
                critic_loss = self.critic_loss(real_proba, fake_proba, loss_to_use)
            else:
                critic_loss = self.critic_loss(fake_proba, real_proba, loss_to_use)

            if use_gp_loss:
                gp = self.gradient_penalty(real_image_stack_batch, real_proba)
                critic_loss = critic_loss + gp
            critic_loss = critic_loss / (factor_critic * self.grad_acc_every)
            critic_loss.backward()

        self.critic_opt.step()
        self.critic.eval()

        self.gen.train()
        self.map_net.train()
        self.gen_opt.zero_grad()

        for _ in range(self.grad_acc_every):
            image_batch, image_cond_batch, part_only_batch = [item.cuda() for item in next(self.loader_G)]
            image_partial_batch = image_cond_batch[:, -1:, :, :]

            latent_vector, cond_feat_maps = self.map_net(image_cond_batch)
            if use_sep_ln:
                latent_vector = None
                noise = self.noise(self.batch_size, self.kwargs['latent-dim'])
                latent_vector = self.style_vec(noise)

            gen_imgs = self.gen(latent_vector, alpha, steps, cond_feature_maps=cond_feat_maps) # steps will be updated in the wrapper function based on the returned value of alpha

            image_batch = self.downsample(image_batch, to_size=gen_imgs.size(-1))
            image_cond_batch = self.downsample(image_cond_batch, to_size=gen_imgs.size(-1))
            part_only_batch = self.downsample(part_only_batch, to_size=gen_imgs.size(-1))
            image_partial_batch = self.downsample(image_partial_batch, to_size=gen_imgs.size(-1))

            gen_imgs_add = torch.max(gen_imgs, image_partial_batch)

            generated_image_stack_batch = torch.cat([image_cond_batch[:, :self.partid], torch.max(gen_imgs, image_cond_batch[:, self.partid:self.partid+1]),
                                                    image_cond_batch[:, self.partid+1:-1], gen_imgs_add], 1)

            gen_fake_proba = self.critic(generated_image_stack_batch, alpha, steps)
            gen_loss = self.gen_loss(gen_fake_proba, gen_imgs, part_only_batch, use_sparsity_loss=self.kwargs['use-sparsity-loss'], sparsity_loss_imp=self.kwargs['sparsity-loss-imp'], loss_to_use=loss_to_use)
            gen_loss = gen_loss / self.grad_acc_every
            gen_loss.backward()

        self.gen_opt.step()
        self.critic.train()
        self.gen.eval()
        self.map_net.eval()

        self.train_step_counter += 1

        return {'gen-loss': gen_loss, 'critic-loss': critic_loss, 'real-prob': real_proba.clone().detach(), 'fake-prob': gen_fake_proba.clone().detach()}

    def critic_loss(self, fake_proba, real_proba, loss_to_use):
        # proba shape == (batch_size, 1) - the final dimension gives prob of reality
        # Employing modified Wasserstein loss
        # return -(torch.mean(real_proba) - torch.mean(fake_proba))
        # return (nn.functional.relu(1 + real_proba) + nn.functional.relu(1 - fake_proba)).mean()

        if loss_to_use == 'wgan':
            return -(torch.mean(real_proba) - torch.mean(fake_proba))
        elif loss_to_use == 'hinge':
            return torch.mean(nn.functional.relu(1-real_proba) + nn.functional.relu(1+fake_proba))

    def gen_loss(self, gen_fake_proba, gen_imgs, real_imgs, use_sparsity_loss, sparsity_loss_imp, loss_to_use):
        # gen_loss = torch.mean(gen_fake_proba) # Apparently doodlerGAN doesn't use the minus!
        if loss_to_use == 'wgan':
            gen_loss = -torch.mean(gen_fake_proba)
        elif loss_to_use == 'hinge':
            gen_loss = -torch.mean(gen_fake_proba)

        if not use_sparsity_loss: return gen_loss

        gen_density = gen_imgs.view(self.batch_size, -1).sum(dim=1)
        real_density = real_imgs.view(self.batch_size, -1).sum(dim=1)
        sparsity_loss = ((gen_density - real_density) ** 2).mean()
        gen_loss = gen_loss + sparsity_loss_imp * sparsity_loss

        return gen_loss

    def print_log(self, train_iter, loss_dict, state_dict):
        base_dir = self.results_dir / '{}/'.format(self.model_name)
        if not os.path.exists(base_dir): os.makedirs(base_dir)

        with open(self.results_dir / '{}/loss.log'.format(self.model_name), 'a') as file:
            file.write('Iteration: {} - Generator loss: {}\t| Discriminator loss: {}\t| Alpha: {}\t|Steps: {}\n'.format(train_iter, loss_dict['gen-loss'], loss_dict['critic-loss'], state_dict['alpha'], state_dict['steps']))

        with open(self.results_dir / '{}/real_proba.log'.format(self.model_name), 'a') as file:
                file.write(str(torch.sigmoid(loss_dict['real-prob']).view(-1)))

        with open(self.results_dir / '{}/fake_proba.log'.format(self.model_name), 'a') as file:
                file.write(str(torch.sigmoid(loss_dict['fake-prob']).view(-1)))

    def train(self, use_sep_ln, flip_p, use_gp_loss, use_label_noise, loss_to_use):
        self.init_critic()
        self.init_gen()
        self.set_data_src()

        model_num = 1
        init_alpha = 1e-5

        alpha = init_alpha
        steps = 0
        max_steps = int(np.log2(self.img_size/4)) # init_img_size = 4
        alpha_one_till = self.kwargs['introduce-layer-after']
        assert alpha_one_till <= 50, 'Fade in shouldn\'t be separated by more than 50 batches'
        alpha_one_ctr = 0
        for train_iter in tqdm(range(self.kwargs['num-train-steps'] - self.grad_acc_every), desc='Training on samples'):
            loss_dict = self.train_step(alpha, steps, use_sep_ln, flip_p, use_gp_loss, use_label_noise, loss_to_use)

            if train_iter % self.alpha_update_every == 0 and train_iter > 0:
                # Update alpha
                alpha = alpha + self.kwargs['alpha-inc']
                alpha = min(alpha, 1)

            if alpha == 1:
                alpha_one_ctr += 1
    
            if alpha_one_ctr == alpha_one_till:
                alpha = init_alpha
                alpha_one_ctr = 0
                steps += 1
                steps = min(max_steps, steps)

            if train_iter % self.kwargs['save-every'] == 0 and train_iter > 0:
                # save generator mapping net
                map_net_base_dir = self.models_dir / 'map_net/{}'.format(self.model_name)
                map_net_path = self.models_dir / 'map_net/{}/model_{}.pt'.format(self.model_name, model_num)
                self.save_model(self.map_net, map_net_path, map_net_base_dir)
                # save generator
                gen_base_dir = self.models_dir / 'generator/{}'.format(self.model_name)
                gen_path = self.models_dir / 'generator/{}/model_{}.pt'.format(self.model_name, model_num)
                self.save_model(self.gen, gen_path, gen_base_dir)

                model_num += 1

            if train_iter % 50 == 0:
                self.print_log(train_iter, loss_dict, {'alpha': alpha, 'steps': steps})

            if train_iter % 1000 == 0 or (train_iter%250 == 0 and train_iter < 2500):
                self.evaluate(num=train_iter, alpha=alpha, steps=steps, use_sep_ln=use_sep_ln)

        # save generator mapping net
        map_net_base_dir = self.models_dir / 'map_net/{}'.format(self.model_name)
        map_net_path = self.models_dir / 'map_net/{}/model_{}.pt'.format(self.model_name, model_num)
        self.save_model(self.map_net, map_net_path, map_net_base_dir)
        # save generator
        gen_base_dir = self.models_dir / 'generator/{}'.format(self.model_name)
        gen_path = self.models_dir / 'generator/{}/model_{}.pt'.format(self.model_name, model_num)
        self.save_model(self.gen, gen_path, gen_base_dir)

        self.print_log(train_iter, loss_dict, {'alpha': alpha, 'steps': steps})

    def save_model(self, model, path, base_dir):
        if not os.path.exists(base_dir): os.makedirs(base_dir)
        torch.save(model.state_dict(), path)

    @torch.no_grad()
    def evaluate(self, num, alpha, steps, use_sep_ln, num_img_tiles=8):
        self.gen.eval()
        self.map_net.eval()

        ext = 'png'
        num_rows = num_img_tiles
        image_batch, image_cond_batch, part_only_batch = [item.cuda() for item in self.dataset_G.sample_partial_test(num_rows ** 2)]
        image_partial_batch = image_cond_batch[:, -1:, :, :]

        latent_vector, cond_feat_maps = self.map_net(image_cond_batch)
        if use_sep_ln:
                latent_vector = None
                noise = self.noise(image_cond_batch.size(0), self.kwargs['latent-dim'])
                latent_vector = self.style_vec(noise)

        gen_imgs = self.gen(latent_vector, alpha, steps, cond_feature_maps=cond_feat_maps)

        image_batch = self.downsample(image_batch, to_size=gen_imgs.size(-1))
        image_cond_batch = self.downsample(image_cond_batch, to_size=gen_imgs.size(-1))
        part_only_batch = self.downsample(part_only_batch, to_size=gen_imgs.size(-1))
        image_partial_batch = self.downsample(image_partial_batch, to_size=gen_imgs.size(-1))

        partial_rgb = self.to_rgb(image_partial_batch, self.default_color)
        part_rgb = part_rgb = self.to_rgb(part_only_batch, self.color)

        base_dir = self.results_dir / self.model_name / 'part-photos'
        if not os.path.exists(base_dir): os.makedirs(base_dir)

        torchvision.utils.save_image(partial_rgb, str(self.results_dir / self.model_name / 'part-photos' / f'{str(num)}-part.{ext}'), nrow=num_rows)
        torchvision.utils.save_image(part_rgb, str(self.results_dir / self.model_name / 'part-photos' / f'{str(num)}-real.{ext}'), nrow=num_rows)
        torchvision.utils.save_image(1-((1-part_rgb)+(1-partial_rgb).clamp_(0., 1.)), str(self.results_dir / self.model_name / 'part-photos' / f'{str(num)}-full.{ext}'), nrow=num_rows)

        generated_part_rgb = self.to_rgb(gen_imgs, self.color)
        torchvision.utils.save_image(generated_part_rgb, str(self.results_dir / self.model_name / 'part-photos' / f'{str(num)}-comp.{ext}'), nrow=num_rows)
        torchvision.utils.save_image(1-((1-generated_part_rgb)+(1-partial_rgb).clamp_(0., 1.)), str(self.results_dir / self.model_name / 'part-photos' / f'{str(num)}.{ext}'), nrow=num_rows)

        self.map_net.train()
        self.gen.eval()

    def to_rgb(self, img, color):
        image_rgb = img.repeat(1, 3, 1, 1)
        return 1-image_rgb*color

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../../data')
    parser.add_argument("--results_dir", type=str, default='../../results')
    parser.add_argument("--models_dir", type=str, default='../../models')
    parser.add_argument('--model_name', type=str, default='default')
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--grad_acc_every', type=int, default=24)
    parser.add_argument('--alpha_update_every', type=int, default=5)
    parser.add_argument('--image_channels', type=int, default=13)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--learning_rate_D', type=float, default=1e-4)
    parser.add_argument('--learning_rate_G', type=float, default=1e-4)

    parser.add_argument('--large_aug', dest='large_aug', action='store_true')
    parser.add_argument('--no-large_aug', dest='large_aug', action='store_false')
    parser.set_defaults(large_aug=False)

    parser.add_argument('--use_sparsity_loss', dest='use_sparsity_loss', action='store_true')
    parser.add_argument('--no-use_sparsity_loss', dest='use_sparsity_loss', action='store_false')
    parser.set_defaults(use_sparsity_loss=False)
    parser.add_argument('--alpha_inc', type=float, default=1e-3)
    parser.add_argument('--sparsity_loss_imp', type=float, default=0.5)
    parser.add_argument('--num_train_steps', type=int, default=50000)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--introduce_layer_after', type=int, default=8)
    
    parser.add_argument('--use_sep_ln', dest='use_sep_ln', action='store_true')
    parser.add_argument('--no-use_sep_ln', dest='use_sep_ln', action='store_false')
    parser.set_defaults(use_sep_ln=False)
    parser.add_argument('--flip_p', type=float, default=0.1)

    parser.add_argument('--use_gp_loss', dest='use_gp_loss', action='store_true')
    parser.add_argument('--no-use_gp_loss', dest='use_gp_loss', action='store_false')
    parser.set_defaults(use_gp_loss=False)

    parser.add_argument('--use_label_noise', dest='use_label_noise', action='store_true')
    parser.add_argument('--no-use_label_noise', dest='use_label_noise', action='store_false')
    parser.set_defaults(use_label_noise=False)
    parser.add_argument('--loss_to_use', type=str, default='hinge')

    args = parser.parse_args()

    assert 0 <= args.flip_p <= 0.2, 'Flipping probability can lie in the range [0, 0.2]'

    kwargs = {'img-channels': args.image_channels, 'latent-dim': args.latent_dim, 'lr-g': args.learning_rate_G,
            'lr-d': args.learning_rate_D, 'large-augmentation': args.large_aug, 'use-sparsity-loss': args.use_sparsity_loss,
            'sparsity-loss-imp': args.sparsity_loss_imp, 'alpha-inc': args.alpha_inc, 'introduce-layer-after': args.introduce_layer_after,
            'num-train-steps': args.num_train_steps, 'save-every': args.save_every, 'model-name': args.model_name}

    trainer = Trainer(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        models_dir=args.models_dir,
        img_size=args.image_size,
        batch_size=args.batch_size,
        grad_acc_every=args.grad_acc_every,
        alpha_update_every=args.alpha_update_every,
        **kwargs
    )

    trainer.train(args.use_sep_ln, args.flip_p, args.use_gp_loss, args.use_label_noise, args.loss_to_use)