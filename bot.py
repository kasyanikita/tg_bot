import torch
import telebot
from bot_token import token
from diffusers import DiffusionPipeline
from io import BytesIO

bot = telebot.TeleBot(token)

@bot.message_handler(commands=['start', 'help'])
def start_message(message):
    bot.send_message(message.from_user.id, "Enter your prompt for image generation")


@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    image = pipe(prompt=message.text, width=512, height=512).images[0]
    bio = BytesIO()
    bio.name = 'image.jpeg'
    image.save(bio, 'JPEG')
    bio.seek(0)
    bot.send_photo(message.from_user.id, photo=bio)


pipe = DiffusionPipeline.from_pretrained(
    "playgroundai/playground-v2-512px-base",
    torch_dtype=torch.float16,
    use_safetensors=True,
    add_watermarker=False,
    variant="fp16",
)
pipe.to("cuda")


bot.infinity_polling()