from typing import Tuple

def early_stopping(val_losses: list[float], patience: int, min_delta: float) -> Tuple[int, int]:
    
    best_loss = val_losses[0]
    best_epoch = 0
    cooldown = 0

    for i in range(1, len(val_losses)):
        
        diff = best_loss - val_losses[i]

        if diff > min_delta:
            best_loss = val_losses[i]
            best_epoch = i
            cooldown = 0
        else:
            cooldown += 1
            
            if cooldown == patience: 
                return (i, best_epoch)

    return (len(val_losses) - 1, best_epoch)