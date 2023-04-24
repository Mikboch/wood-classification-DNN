from src.models.train_model import train_model


def main():
    model, results = train_model(
        model_name="WIMCNNModel",
        model_hparams={"num_classes": 12, "c_hidden": [16, 32, 64], "num_blocks": [3, 3, 3], "act_fn_name": "relu"},
        optimizer_name="SGD",
        optimizer_hparams={"lr": 0.1, "momentum": 0.9, "weight_decay": 1e-4},
    )
    print("Results: ", results)


if __name__ == '__main__':
    main()
