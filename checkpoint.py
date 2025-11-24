import torch


def save_checkpoint(path, model, optimizer, epoch, global_steps, tokens_seen, best_val_loss):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_steps": global_steps,
        "tokens_seen": tokens_seen,
        "best_val_loss": best_val_loss,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint["epoch"]
    global_steps = checkpoint["global_steps"]
    tokens_seen = checkpoint["tokens_seen"]
    best_val_loss = checkpoint["best_val_loss"]

    print(f"[CHECKPOINT] Carregado de: {path} (epoch {epoch}, step {global_steps})")

    return epoch, global_steps, tokens_seen, best_val_loss