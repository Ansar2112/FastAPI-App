from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch

# Path to your local model
model_path = "models/Qwen2-VL-2B-Instruct"

# Load processor and model from local directory
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Move to GPU
# model.to("cuda")

# Load your image
image = Image.open("Invoice.png")  # replace with your image path

# User prompt
prompt = "Extract all table data from this invoice in JSON format"

# Prepare input
inputs = processor(text=[prompt], images=[image], return_tensors="pt").to("cuda")

# Generate output
outputs = model.generate(**inputs, max_new_tokens=500)

# Decode output
result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
print("\n🧾 Extracted Data:\n", result)
