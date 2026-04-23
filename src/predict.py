model = TextCNN(
    vocab_size=config["vocab_size"],
    embed_dim=config["embed_dim"],
    num_classes=config["num_classes"],
    num_filters=config["num_filters"],
    kernel_sizes=config["kernel_sizes"],
    dropout=config["dropout"],
    pad_idx=config["pad_idx"],
).to(device)