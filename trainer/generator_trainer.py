import torch


class GeneratorTrainer:
    def __init__(self, generator, critic, batch_size, train_num, lr, betas, decay_step, decay_rate):
        self.generator = generator
        self.critic = critic
        self.batch_size = batch_size
        self.train_num = train_num

        self.code_num = self.generator.code_num
        self.optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=decay_step, gamma=decay_rate)
        self.device = self.generator.device

    def _step(self, target_codes, lens):
        noise = self.generator.get_noise(len(lens))  # [B, code_num]
        samples, hiddens = self.generator(target_codes, lens, noise)  # samples: [B, T, C], hiddens: [B, T, H]
        output = self.critic(samples, hiddens, lens)
        loss = -output.mean()

        self.optimizer.zero_grad()
        loss.backward()

        # Optional for Transformer: gradient clipping (to avoid exploding gradients)
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)

        self.optimizer.step()
        return loss

    def step(self, target_codes, lens):
        self.generator.train()
        self.critic.eval()

        total_loss = 0
        for _ in range(self.train_num):
            total_loss += self._step(target_codes, lens).item()
        avg_loss = total_loss / self.train_num
        self.scheduler.step()

        return avg_loss
