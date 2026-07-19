from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
model_type = "vit_t"
sam_checkpoint = "mobile_sam.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# mobile_sam.to(device=device)
# mobile_sam.eval()




