import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Create a tensor and transpose it (this results in a non-contiguous tensor)
x = torch.randn(2, 3).to(device)

if torch
y = (
    x.T
)  # Transpose, which results in a non-contiguous tensor but shares the same memory address
print(x.shape, x.stride())

# Check if the tensor is contiguous
print(x.is_contiguous())  # True
print(y.is_contiguous())  # False
print(y.shape, y.stride())
# Use .contiguous() to make the tensor contiguous
z = y.contiguous()
print(z.is_contiguous())  # True

# You can now safely use .view() or .reshape() on z
z_reshaped = z.view(6)  # This will work now
