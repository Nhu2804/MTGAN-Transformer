import torch

from model import PredictNextLoss


class BaseTransformerTrainer:
    def __init__(self, args, base_transformer, max_len, train_loader, params_path):
        self.base_transformer = base_transformer
        self.train_loader = train_loader
        self.params_path = params_path

        self.epochs = args.base_transformer_epochs
        self.optimizer = torch.optim.Adam(base_transformer.parameters(), lr=args.base_transformer_lr)
        self.loss_fn = PredictNextLoss(max_len)

    def train(self):
        print('Pre-training base transformer...')
        for epoch in range(1, self.epochs + 1):
            print('Epoch %d / %d:' % (epoch, self.epochs))
            total_loss = 0.0
            total_num = 0
            steps = len(self.train_loader)

            for step, data in enumerate(self.train_loader, start=1):
                x, lens, y = data  # x: [B, T, code_num]
                output = self.base_transformer(x)  # output: [B, T, code_num]
                loss = self.loss_fn(output, y, lens)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * len(x)
                total_num += len(x)

                print('\r    Step %d / %d, loss: %.4f' % (step, steps, total_loss / total_num), end='')

            print('\r    Step %d / %d, loss: %.4f' % (steps, steps, total_loss / total_num))

        self.base_transformer.save(self.params_path)
