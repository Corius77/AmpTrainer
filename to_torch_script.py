import torch
import myk_models

name = 'ht1_from_book'
saved_pth_path = f"{name}.pth"
export_pt_path = f"{name}.pt"

model = torch.load(saved_pth_path, weights_only=False, map_location='cpu')
model.eval()

scripted_model = torch.jit.script(model)
torch.jit.save(scripted_model, export_pt_path)
