from flask import Flask, render_template, request, send_file, abort
import torch
from diffusers import StableDiffusionPipeline
from io import BytesIO
import logging

# Modelin başlangıçta bir kez yüklenmesi
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")  # GPU kullanımı için

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    text = request.form['text']
    guidance_scale = float(request.form.get('guidance_scale', 7.5))  # Varsayılan değer 7.5
    num_inference_steps = int(request.form.get('num_inference_steps', 50))  # Varsayılan değer 50
    negative_prompt = request.form.get('negative_prompt', '')

    try:
        # Görüntü üretimi
        image = pipe(text, negative_prompt=negative_prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images[0]
    except Exception as e:
        logger.error(f"Model hatası: {e}")
        abort(500, description="Görsel üretim sırasında bir hata oluştu.")
    
    # Görüntüyü hafızada tutarak BytesIO ile dosya olarak göndermek
    img_io = BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)
    
    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    from waitress import serve
    serve(app, host='127.0.0.1', port=5000)
