from src.data.make_dataset import get_data_loaders
from src.models.net.w_imcnn_model import WIMCNNModel
from src.models.train_model import train_model


def main():
    train_loader, val_loader = get_data_loaders()
    model = WIMCNNModel()
    model, results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_name="SGD",
        optimizer_hparams={"lr": 0.1, "momentum": 0.9, "weight_decay": 1e-4},
    )
    print("Results: ", results)


if __name__ == '__main__':
    main()
