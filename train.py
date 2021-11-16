from data_loader import Dataset_JSON
from torch.utils import data
import torch
from torch.optim import Adam
import multiprocessing
from pathlib import Path
from discriminator import Discriminator
from generator import Generator, MappingNetwork
import tqdm
import argparse

def cycle(iterable):
    while True:
        for i in iterable:
            yield i

class Trainer:
    def __init__(self, data_dir, results_dir, models_dir, img_size, batch_size, grad_acc_every, alpha_update_every, **kwargs):
        assert alpha_update_every < grad_acc_every, 'Alpha should be updated atleast once in a batch'
        assert alpha_update_every <= 10, 'Alpha should be updated atleast once in 10 steps'

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

    def init_gen(self):
        self.map_net = MappingNetwork(in_size=self.img_size, in_ch=self.kwargs['img-channels'], latent_dim=self.kwargs['latent-dim'])
        self.gen = Generator(self.img_size, self.kwargs['latent-dim'], reversed(self.map_net.channels), img_channels=self.kwargs['img-channels'], concat_map=True)
        gen_params = list(self.map_net.parameters()) + list(self.gen.parameters())
        self.gen_opt = Adam(gen_params, lr=self.kwargs['lr-g'], betas=(0., 0.99))

    def init_critic(self):
        self.critic = Discriminator(self.img_size, img_channels=self.kwargs['img-channels'])
        self.critic_opt = Adam(self.critic.parameters(), lr=self.kwargs['lr-d'], betas=(0., 0.99))

    def set_data_src(self):
        data_dir = self.data_dir
        large_aug = self.kwargs['large-augmentation']
        num_cores = multiprocessing.cpu_count()
        self.dataset_D = Dataset_JSON(data_dir, self.img_size, large_aug=large_aug)
        self.dataset_G = Dataset_JSON(data_dir, self.img_size, large_aug=large_aug)
        self.loader_D = cycle(data.DataLoader(self.dataset_D, num_workers = int(num_cores/2), batch_size = self.batch_size, drop_last = True, shuffle=True, pin_memory=True))
        self.loader_G = cycle(data.DataLoader(self.dataset_G, num_workers = int(num_cores/2), batch_size = self.batch_size, drop_last = True, shuffle=True, pin_memory=True))

    def train_step(self, alpha, steps):
        self.critic_opt.zero_grad()
        self.gen_opt.zero_grad()

        for _ in range(self.grad_acc_every):
            # TODO: Get the data
            image_batch, image_cond_batch, part_only_batch = [item.cuda() for item in next(self.loader_D)]

            z_latent = torch.randn(self.batch_size, self.kwargs['latent-dim'])
            # TODO: Update the forward call to the mapping network
            latent_vector = self.map_net(z_latent)
            gen_imgs = self.gen(latent_vector, alpha, steps) # steps will be updated in the wrapper function based on the returned value of alpha

            fake_proba = self.critic(gen_imgs.clone().detach(), alpha, steps)
            real_image_stack_batch = torch.cat([image_cond_batch[:, :self.partid], torch.max(part_only_batch, image_cond_batch[:, self.partid:self.partid+1]),
                                                        image_cond_batch[:, self.partid+1:-1], image_batch], 1)
            real_image_stack_batch.requires_grad_()
            real_proba = self.critic(real_image_stack_batch)
            
            critic_loss = self.critic_loss(fake_proba, real_proba)
            critic_loss = critic_loss / self.grad_acc_every
            critic_loss.backward()

            gen_fake_proba = self.critic(gen_imgs, alpha, steps)
            gen_loss = self.gen_loss(gen_fake_proba, gen_imgs, part_only_batch, use_sparsity_loss=self.kwargs['use-sparsity-loss'], sparsity_loss_imp=self.kwargs['sparsity-loss-imp'])
            gen_loss = gen_loss / self.grad_acc_every
            gen_loss.backward()

            if _ % self.alpha_update_every == 0:
                # Update alpha
                alpha = alpha + self.kwargs['alpha-inc']
                alpha = min(alpha, 1)

        self.critic_opt.step()
        self.gen_opt.step()

        self.train_step_counter += 1

        return alpha, {'gen-loss': gen_loss, 'critic-loss': critic_loss}

    def critic_loss(self, fake_proba, real_proba):
        # proba shape == (batch_size, 1) - the final dimension gives prob of reality
        # Employing Wasserstein loss
        return -torch.mean(real_proba - fake_proba)

    def gen_loss(self, gen_fake_proba, gen_imgs, real_imgs, use_sparsity_loss, sparsity_loss_imp):
        gen_loss = -torch.mean(gen_fake_proba)
        if not use_sparsity_loss: return gen_loss

        gen_density = gen_imgs.view(self.batch_size, -1).sum(dim=1)
        real_density = real_imgs.view(self.batch_size, -1).sum(dim=1)
        sparsity_loss = ((gen_density - real_density) ** 2).mean()
        gen_loss = gen_loss + sparsity_loss_imp * sparsity_loss

        return gen_loss

    def print_log(self, train_iter, loss_dict):
        with open(self.results_dir + '/loss.log', 'a') as file:
            file.write('Iteration: {} - Generator loss: {} | Discriminator loss: {}\n'.format(train_iter, loss_dict['gen-loss'], loss_dict['critic-loss']))

    def train(self):
        self.init_critic()
        self.init_gen()
        self.set_data_src()

        model_num = 1

        init_alpha = 1e-5

        alpha = init_alpha
        steps = 0
        alpha_one_till = self.kwargs['introduce-layer-after']
        assert alpha_one_till <= 8, 'Fade in shouldn\'t be separated by more than 8 batches'
        alpha_one_ctr = 0
        for train_iter in tqdm(range(self.kwargs['num-train-steps'] - self.grad_acc_every), desc='Training on samples'):
            alpha, loss_dict = self.train_step(alpha, steps)

            if alpha == 1:
                alpha_one_ctr += 1
    
            if alpha_one_ctr == alpha_one_till:
                alpha = init_alpha
                steps += 1

        if train_iter % self.kwargs['save-every']:
            # save generator mapping net
            map_net_path = self.models_dir + '/map_net/{self.model_name}/model_{model_num}.pt'
            self.save_model(self.map_net, map_net_path)
            # save generator
            gen_path = self.models_dir + '/generator/{self.model_name}/model_{model_num}.pt'
            self.save_model(self.gen, gen_path)

            model_num += 1

        if train_iter % 50 == 0:
            self.print_log(train_iter, loss_dict)

    def save_model(model, path):
        torch.save(model.state_dict(), path)

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
    parser.add_argument('--large_aug', type=bool, default=False)
    parser.add_argument('--use_sparsity_loss', type=bool, default=True)
    parser.add_argument('--alpha_inc', type=float, default=1e-3)
    parser.add_argument('--sparsity_loss_imp', type=float, default=0.5)
    parser.add_argument('--num_train_steps', type=int, default=50000)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--introduce_layer_after', type=int, default=8)

    args = parser.parse_args()

    kwargs = {'img-channels': args['image_channels'], 'latent-dim': args['latent_dim'], 'lr-g': args['learning_rate_G'],
            'lr-d': args['learning_rate_D'], 'large-augmentation': args['large_aug'], 'use-sparsity-loss': args['use_sparsity_loss'],
            'sparsity-loss-imp': args['sparsity_loss_imp'], 'alpha-inc': args['alpha_inc'], 'introduce-layer-after': args['introduce_layer_after'],
            'num-train-steps': args['num_train_steps'], 'save-every': args['save_every'], 'model-name': args['model_name']}

    trainer = Trainer(
        data_dir=args['data_dir'],
        results_dir=args['results_dir'],
        models_dir=args['models_dir'],
        img_size=args['image_size'],
        batch_size=args['batch_size'],
        grad_acc_every=args['grad_acc_every'],
        alpha_update_every=args['alpha_update_every'],
        **kwargs
    )