from transformers import AutoModel, AutoTokenizer
import torch
import os


# Dynamic GPU allocation via orchestrator
def _allocate_gpu_for_ocr():
    """Allocate GPU from orchestrator if available."""
    gpu_orchestrator_url = os.getenv("GPU_ORCHESTRATOR_URL")
    if not gpu_orchestrator_url:
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        return [0]
    
    try:
        import httpx
        request_data = {
            "service_name": "deepseek-ocr",
            "workload_type": "ocr",
            "workload_data": {"image_count": 1}
        }
        
        response = httpx.post(
            f"{gpu_orchestrator_url}/gpu/allocate",
            json=request_data,
            timeout=10.0
        )
        
        if response.status_code == 200:
            allocation = response.json()
            device_ids = allocation.get("gpu_ids", [0])
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, device_ids))
            return device_ids
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            return [0]
    except Exception as e:
        print(f"Warning: Failed to allocate GPU from orchestrator: {e}, using GPU 0")
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        return [0]

_allocated_gpus = _allocate_gpu_for_ocr()


model_name = 'deepseek-ai/DeepSeek-OCR'


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True)
model = model.eval().cuda().to(torch.bfloat16)



# prompt = "<image>\nFree OCR. "
prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = 'your_image.jpg'
output_path = 'your/output/dir'



# infer(self, tokenizer, prompt='', image_file='', output_path = ' ', base_size = 1024, image_size = 640, crop_mode = True, test_compress = False, save_results = False):

# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False

# Gundam: base_size = 1024, image_size = 640, crop_mode = True

res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, base_size = 1024, image_size = 640, crop_mode=True, save_results = True, test_compress = True)
