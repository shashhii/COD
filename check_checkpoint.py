import torch

checkpoint_path = r"C:\Users\shash\Downloads\COD\COD10K Trained model\COD10K_best.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("Checkpoint keys:", checkpoint.keys())
print("\nModel state dict keys (first 10):")
state_dict = checkpoint['model_state_dict']
for i, key in enumerate(list(state_dict.keys())[:10]):
    print(f"  {key}: {state_dict[key].shape}")

print(f"\nTotal parameters: {len(state_dict)}")
print(f"\nHas _orig_mod prefix: {any(k.startswith('_orig_mod.') for k in state_dict.keys())}")
