import torch
import numpy as np

from src.config.config import cfg


class Trainer:
    def __init__(self, model, criterion, optimizer):
        self.model = model.to(cfg.DEVICE)
        self.criterion = criterion
        self.optimizer = optimizer

    def train_step(self, train_loader):
        self.model.train()
        self.model.to(cfg.DEVICE)
        train_loss = 0
        for data in train_loader:
            data = data.to(cfg.DEVICE)
            self.optimizer.zero_grad()

            output = self.model(data)
            loss = self.criterion(output, data)

            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        return avg_train_loss

    def validation_step(self, validation_loader):
        self.model.to("cpu")
        val_loss = 0
        with torch.no_grad():
            for data in iter(validation_loader):
                data = data.to("cpu")
                prediction = self.model(data)
                loss = self.criterion(prediction, data)
                # cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                # cosine = cos(data, prediction).tolist()
                # batch_pred = np.where(np.array(cosine) < 0.95, 1, 0).tolist()

                val_loss += loss.item()
            avg_validation_loss = val_loss / len(validation_loader)

        return avg_validation_loss

    def fit(self, train_loader, val_loader):
        for epoch in range(cfg.EPOCHS):
            train_loss = self.train_step(train_loader)
            val_loss = self.validation_step(val_loader)

            print(f"Epoch [{epoch + 1}/{cfg.EPOCHS}]"
                  f"Training Loss: {train_loss:.7f} "
                  f"Validation Loss: {val_loss:.7f} "
                  )


def evaluation(test_loader, model):
    model.eval()
    model.to("cpu")
    pred = []
    with torch.no_grad():
        for data in iter(test_loader):
            data = data.to("cpu")
            prediction = model(data)

            # cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            # cosine = cos(data, prediction).tolist()
            # batch_pred = np.where(np.array(cosine) < 0.95, 1, 0).tolist()

            mse = np.mean(np.power(data.detach().numpy() - prediction.detach().numpy(), 2), axis=1)
            threshold = np.mean(mse) + 3 * np.std(mse)
            batch_pred = np.where(np.array(mse) < threshold, 0, 1).tolist()
            pred += batch_pred

    return pred
