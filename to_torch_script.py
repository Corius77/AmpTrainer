import torch
import myk_models

name = 'orange'
saved_pth_path = "lstm_size_32_epoch_1321_loss_0.0692.pth"
export_pt_path = f"{name}.pt"

model = torch.load(saved_pth_path, weights_only=False, map_location='cpu')
model.eval()

scripted_model = torch.jit.script(model)
torch.jit.save(scripted_model, export_pt_path)
