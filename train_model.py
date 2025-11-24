from config import cosine_decay
from generate import generate_text
from evaluate import evaluate_model, plot_losses
from batch_config import token_ids_to_text,calc_loss_batch
from checkpoint import save_checkpoint
import torch 

def train(
        model,
        train_loader,
        val_loader,
        eval_freq,
        eval_iter,
        optimizer,
        device,
        num_epochs,
        temperature,
        max_new_tokens,
        context_size,
        top_k,
        sample,
        tokenizer,
        checkpoint_path="checkpoint.pth",
        resume=False
    ):

        train_losses, val_losses, tokens_seens = [], [], []

        global_steps, tokens_seen = -1, 0
        best_val_loss = float("inf")
        start_epoch = 0

        for epoch in range(start_epoch, num_epochs):
            model.train()

            for i, (xb, yb) in enumerate(train_loader):
                optimizer.zero_grad(set_to_none = True)

                loss = calc_loss_batch(model, xb, yb, device)
                loss.backward()

                global_steps += 1
                tokens_seen += xb.numel()

                lr = cosine_decay(global_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                if global_steps % eval_freq == 0:
                    train_losse, val_losse = evaluate_model(
                        model, train_loader, val_loader, eval_iter, device
                    )

                    train_losses.append(train_losse)
                    val_losses.append(val_losse)
                    tokens_seens.append(tokens_seen)

                    print(f"Steps: {global_steps+1} | Epoch: {epoch+1} | "
                          f"Train: {train_losse:.3f} | Val: {val_losse:.3f} | "
                          f"Tokens: {tokens_seen} | Norm: {norm:.3f} | LR: {lr}")

                    if val_losse < best_val_loss:
                        best_val_loss = val_losse
                        save_checkpoint(
                            "best_model.pth", model, optimizer,
                            epoch, global_steps, tokens_seen, best_val_loss
                        )

                    save_checkpoint(
                        checkpoint_path, model, optimizer,
                        epoch, global_steps, tokens_seen, best_val_loss
                    )

            out = generate_text(model, sample, context_size, max_new_tokens, temperature, top_k)
            print(token_ids_to_text(out, tokenizer))

        epochs_tensor = torch.linspace(0 , num_epochs, len(train_losses))
        plot_losses(epochs_tensor,tokens_seens, train_losses, val_losses)
        print(len(tokens_seens))
        torch.save(model.state_dict(), "model_final.pth")

        return train_losses, val_losses, tokens_seens





