import torch.onnx
from networks import DeepQNetwork, DuelingDeepQNetwork


def convert_onnx(model, name, input_size):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.eval()
    dummy_input = torch.randn(1, input_size, requires_grad=True).to(device)

    # Export the model
    torch.onnx.export(
        model,
        dummy_input,
        f"data/{name}.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print("Model exported to ONNX format")


if __name__ == '__main__':
    input_size = 37
    n_actions = 4

    deep_q_model = DeepQNetwork(lr=0.0001, n_actions=n_actions,
                                input_dims=input_size,
                                name="deep_q_model",
                                checkpoint_dir="")

    dueling_deep_q_model = DuelingDeepQNetwork(lr=0.0001, n_actions=n_actions,
                                input_dims=input_size,
                                name="dueling_deep_q_model",
                                checkpoint_dir="")

    convert_onnx(deep_q_model, "DeepQNetwork", input_size)
    convert_onnx(dueling_deep_q_model, "DuelingDeepQNetwork", input_size)