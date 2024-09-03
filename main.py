import tiktoken
import asyncio
from database import DB
from aiogram import Bot, Dispatcher, Router
from aiogram.filters.command import Command
from aiogram.filters.callback_data import CallbackData, CallbackQueryFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.filters.state import StateFilter
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram import F
from aiogram.types import Message, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery, LabeledPrice, PreCheckoutQuery, BufferedInputFile
import os
from dotenv import load_dotenv
import orjson
import openai
import aiohttp
import traceback
from base64 import b64decode
import time

load_dotenv()

BOT_TOKEN = os.getenv('BOT_TOKEN')
VSEGPT_TOKEN = os.getenv('VSEGPT_TOKEN')
VSEGPT_URL = os.getenv('VSEGPT_URL')
CHAT_MAIN_MODEL = os.getenv('CHAT_MAIN_MODEL')
ADMIN_ID = os.getenv('ADMIN_ID')
CHAT_TEMPERATURE = float(os.getenv('CHAT_TEMPERATURE'))
CHAT_MAX_WORDS = int(os.getenv('CHAT_MAX_WORDS'))
ENABLE_OMF = os.getenv('ENABLE_OMF') in ['1', 'TRUE', 'YES']
DEVELOPER_INFO = os.getenv('DEVELOPER_INFO')
TERMS_OF_USE = os.getenv('TERMS_OF_USE')
PRIVACY_POLICY = os.getenv('PRIVACY_POLICY')
GUIDE = os.getenv('GUIDE')
TEST_BALANCE = int(os.getenv('TEST_BALANCE'))

# Debugging prints
print(f"BOT_TOKEN: {BOT_TOKEN}")
print(f"VSEGPT_TOKEN: {VSEGPT_TOKEN}")
print(f"VSEGPT_URL: {VSEGPT_URL}")
print(f"CHAT_MAIN_MODEL: {CHAT_MAIN_MODEL}")
print(f"ADMIN_ID: {ADMIN_ID}")
print(f"CHAT_TEMPERATURE: {CHAT_TEMPERATURE}")
print(f"CHAT_MAX_WORDS: {CHAT_MAX_WORDS}")
print(f"ENABLE_OMF: {ENABLE_OMF}")
print(f"DEVELOPER_INFO: {DEVELOPER_INFO}")
print(f"TERMS_OF_USE: {TERMS_OF_USE}")
print(f"PRIVACY_POLICY: {PRIVACY_POLICY}")
print(f"GUIDE: {GUIDE}")
print(f"TEST_BALANCE: {TEST_BALANCE}")

with open('models.json', 'r') as f:
    MODELS = orjson.loads(f.read())

IMAGE_MODEL = MODELS['recommended']['txt2img']['cheap']

QUEUED_USERS = []

openai_client = openai.OpenAI(
    api_key=VSEGPT_TOKEN,
    base_url=VSEGPT_URL,
    max_retries=5,
    timeout=20.0
)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())
db = DB('data.sqlite')

class Settings(StatesGroup):
    model = State()
    temperature = State()
    max_words = State()
    scenario = State()

class MakeScenario(StatesGroup):
    scenario_name = State()
    scenario_description = State()
    example_dialogues = State()

model_pricing_cache = {}
models_list_cache = {}

async def get_models_list():
    if 'models_list' in models_list_cache:
        return models_list_cache['models_list']
    models_list = await openai_client.models.list()
    models_list_cache['models_list'] = models_list
    return models_list

async def get_model_pricing(model_name):
    if model_name in model_pricing_cache:
        return model_pricing_cache[model_name]
    
    models_list = await get_models_list()
    for model in models_list:
        if model['id'] == model_name:
            pricing = model['pricing']
            model_pricing_cache[model_name] = pricing
            return pricing
    
    default_pricing = {
        'prompt': 0.1,
        'completion': 0.15
    }
    model_pricing_cache[model_name] = default_pricing
    return default_pricing

# Добавляем кэш для энкодеров tiktoken
tiktoken_encoders = {}

def get_tiktoken_encoder(model):
    if model not in tiktoken_encoders:
        tiktoken_encoders[model] = tiktoken.encoding_for_model(model)
    return tiktoken_encoders[model]

def count_tokens(text, model):
    encoder = get_tiktoken_encoder(model)
    return len(encoder.encode(text))

async def stream_response(message: Message, response_stream, model, edit_interval=3):
    full_response = ""
    last_edit_time = time.time()
    sent_message = None
    total_tokens = 0

    async for chunk in response_stream:
        if chunk.choices[0].delta.content is not None:
            new_content = chunk.choices[0].delta.content
            full_response += new_content
            total_tokens += count_tokens(new_content, model)
            
            current_time = time.time()
            if current_time - last_edit_time >= edit_interval:
                if sent_message:
                    try:
                        await sent_message.edit_text(full_response[:4096])
                    except:
                        pass
                else:
                    sent_message = await message.answer(full_response[:4096])
                last_edit_time = current_time

    if sent_message:
        await sent_message.edit_text(full_response[:4096])
    else:
        await message.answer(full_response[:4096])

    return full_response, sent_message, total_tokens

@dp.message(Command('start', 'menu'))
async def start_command(message: Message):
    user_id = message.from_user.id
    user = await db.get_user(user_id)
    if not user:
        await db.add_user(user_id, balance=TEST_BALANCE)
        user = await db.get_user(user_id)
    
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text='Очистить чат', callback_data='clear')],
            [InlineKeyboardButton(text='Настройки', callback_data='settings')],
            [InlineKeyboardButton(text='Создать сценарий', callback_data='make_scenario')],
            [InlineKeyboardButton(text='Пополнить баланс', callback_data='topup')],
            [InlineKeyboardButton(text='Руководство', callback_data='guide')],
        ]
    )
    await message.answer(f'Привет!\nТвой баланс: {round(user["balance"], 2)} кредитов', reply_markup=keyboard)

@dp.message(Command('clear'))
async def clear_command(message: Message):
    if message.from_user.id in QUEUED_USERS:
        await message.answer('Сначала дождитесь выполнения предыдущего запроса.')
        return None
    await db.clear_chat(message.from_user.id)
    await message.answer('Чат очищен')

@dp.message(Command('developer_info'))
async def developer_info_command(message: Message):
    await message.answer(f'Информация о разработчике:\n{DEVELOPER_INFO}')

@dp.message(Command('terms'))
async def terms_of_use_command(message: Message):
    await message.answer(f'Условия использования:\n{TERMS_OF_USE}')

@dp.message(Command('privacy_policy'))
async def privacy_policy_command(message: Message):
    await message.answer(f'Политика конфиденциальности:\n{PRIVACY_POLICY}')

@dp.message(Command('help'))
async def help_command(message: Message):
    await message.answer(f'Руководство:\n{GUIDE}')

@dp.message(Command('image'))
async def image_command(message: Message):
    if message.from_user.id in QUEUED_USERS:
        await message.answer('Сначала дождитесь выполнения предыдущего запроса.')
        return None
    QUEUED_USERS.append(message.from_user.id)
    wait = await message.answer('Подождите немного...')
    prompt = message.text.removeprefix('/image ')
    if not prompt:
        await message.answer('Пожалуйста, введите описание изображения.')
        return None
    user_data = await db.get_user(message.from_user.id)
    if user_data['balance'] <= 1.8 and message.from_user.id != int(ADMIN_ID):
        await message.answer('Недостаточно кредитов на балансе для отправки запроса.\nКупите кредиты в разделе "Пополнить баланс".')
        return None
    await db.decrease_balance(message.from_user.id, 1.8)
    response = openai_client.images.generate(model=IMAGE_MODEL, prompt=prompt, n=1, size='1024x1024', response_format='b64_json')
    image_b64_json = response.data[0].b64_json
    image = b64decode(image_b64_json)
    await message.answer_photo(BufferedInputFile(image, filename=f'image_{time.time()}.png'))
    QUEUED_USERS.remove(message.from_user.id)
    await wait.delete()

@dp.message(Command('cancel'))
async def cancel_command(message: Message):
    if message.from_user.id in QUEUED_USERS:
        await message.answer('Сначала дождитесь выполнения предыдущего запроса.')
        return None
    user_data = await db.get_user(message.from_user.id)
    chat_history = user_data['chat_history']
    user_messages = [msg for msg in chat_history if msg['role'] != 'system']
    if len(user_messages) > 2:
        chat_history = chat_history[:-2]
        await db.update_user(message.from_user.id, {'chat_history': chat_history})
        await message.answer('Последний запрос отменен.')
    else:
        await message.answer('Недостаточно сообщений для удаления.')

@dp.message(Command('reroll'))
async def reroll_command(message: Message):
    if message.from_user.id in QUEUED_USERS:
        await message.answer('Сначала дождитесь выполнения предыдущего запроса.')
        return None
    QUEUED_USERS.append(message.from_user.id)
    wait = await message.answer(text='Подождите немного...')
    user_id = message.from_user.id
    settings = await db.get_settings(user_id)
    user_data = await db.get_user(user_id)
    
    if user_data['balance'] < 0 and user_id != int(ADMIN_ID):
        await message.answer('Недостаточно кредитов на балансе для отправки запроса.\nКупите кредиты в разделе "Пополнить баланс".')
        QUEUED_USERS.remove(user_id)
        return None
    
    chat_history = user_data['chat_history']
    if len(chat_history) < 2 or chat_history[-1]['role'] != 'assistant':
        await message.answer('Нет предыдущего ответа для повторной генерации.')
        QUEUED_USERS.remove(user_id)
        return None
    
    # Удаляем последний ответ ассистента
    chat_history.pop()
    
    model = settings.get('model')
    try:
        response_stream = openai_client.chat.completions.create(
            model=model,
            messages=chat_history,
            temperature=settings.get('temperature'),
            stream=True
        )
        new_response, sent_message, completion_tokens = await stream_response(message, response_stream, model)
    except Exception as e:
        await message.answer('Ошибка при генерации нового ответа!')
        traceback.print_exc()
        QUEUED_USERS.remove(user_id)
        return None
    
    chat_history.append({'role': 'assistant', 'content': new_response})
    
    try:
        model_pricing = await get_model_pricing(model)
    except:
        traceback.print_exc()
        QUEUED_USERS.remove(user_id)
        await message.answer('Ошибка при получении цены модели!')
        return None
    
    prompt_tokens = sum(count_tokens(msg['content'], model) for msg in chat_history if msg['role'] != 'assistant')
    spent_prompt_credits = prompt_tokens * float(model_pricing['prompt']) / 1000
    spent_completion_credits = completion_tokens * float(model_pricing['completion']) / 1000
    spent_credits = spent_prompt_credits + spent_completion_credits
    
    await db.decrease_balance(user_id, spent_credits)
    await db.increase_total_chat_requests(user_id, prompt_tokens + completion_tokens)
    await db.update_user(user_id, {'chat_history': chat_history, 'balance': user_data['balance'], 'settings': settings})
    
    await wait.delete()
    QUEUED_USERS.remove(user_id)

@dp.message(F.text)
async def answer_to_message(message: Message):
    if message.from_user.id in QUEUED_USERS:
        await message.answer('Сначала дождитесь выполнения предыдущего запроса.')
        return None
    QUEUED_USERS.append(message.from_user.id)
    wait = await message.answer(text='Подождите немного...')
    user_id = message.from_user.id
    settings = await db.get_settings(user_id)
    user_data = await db.get_user(user_id)
    if user_data['balance'] < 0 and user_id != int(ADMIN_ID):
        await message.answer('Недостаточно кредитов на балансе для отправки запроса.\nКупите кредиты в разделе "Пополнить баланс".')
        QUEUED_USERS.remove(message.from_user.id)
        return None
    chat_history = user_data['chat_history']
    chat_history.append({"role": "user", "content": message.text})
    
    try:
        contains_image = False
        for msg in chat_history:
            if type(msg['content']) == list:
                for cont in msg['content']:
                    if cont['type'] == 'image_url':
                        contains_image = True
                        break
            if contains_image:
                break
        if contains_image:
            model = settings.get('vision_model')
        else:
            model = settings.get('model')
        
        response_stream = openai_client.chat.completions.create(
            model=model,
            messages=chat_history,
            temperature=settings.get('temperature'),
            stream=True
        )
        response_text, sent_message, completion_tokens = await stream_response(message, response_stream, model)
    except Exception as e:
        await message.answer('Ошибка!')
        traceback.print_exc()
        QUEUED_USERS.remove(message.from_user.id)
        return None
    
    chat_history.append({'role': 'assistant', 'content': response_text})
    max_words = settings.get('max_words')
    total_words = sum(len(msg['content'].split()) if isinstance(msg['content'], str) else sum(len(cont['text'].split()) for cont in msg['content'] if cont['type'] == 'text') for msg in chat_history if msg['role'] != 'system')
    
    while total_words > max_words:
        if chat_history[0]['role'] != 'system':
            if isinstance(chat_history[0]['content'], str):
                total_words -= len(chat_history[0]['content'].split())
            else:
                total_words -= sum(len(cont['text'].split()) for cont in chat_history[0]['content'] if cont['type'] == 'text')
            chat_history.pop(0)
        else:
            chat_history.pop(0)
    await db.update_user(user_id, {'chat_history': chat_history, 'balance': user_data['balance'], 'settings': settings})
    
    try:
        model_pricing = await get_model_pricing(model)
    except:
        traceback.print_exc()
        QUEUED_USERS.remove(message.from_user.id)
        await message.answer('Ошибка!')
        return None
    
    prompt_tokens = sum(count_tokens(msg['content'], model) for msg in chat_history if msg['role'] != 'assistant')
    spent_prompt_credits = prompt_tokens * float(model_pricing['prompt']) / 1000
    spent_completion_credits = completion_tokens * float(model_pricing['completion']) / 1000
    spent_credits = spent_prompt_credits + spent_completion_credits
    
    await db.decrease_balance(user_id, spent_credits)
    await db.increase_total_chat_requests(user_id, prompt_tokens + completion_tokens)
    
    await wait.delete()
    QUEUED_USERS.remove(message.from_user.id)

@dp.message(F.content_type == 'photo')
async def answer_to_image(message: Message):
    if message.from_user.id in QUEUED_USERS:
        await message.answer('Сначала дождитесь выполнения предыдущего запроса.')
        return None
    QUEUED_USERS.append(message.from_user.id)
    wait = await message.answer('Подождите немного...')
    user_id = message.from_user.id
    settings = await db.get_settings(user_id)
    user_data = await db.get_user(user_id)
    if user_data['balance'] < 0 and user_id != int(ADMIN_ID):
        await message.answer('Недостаточно кредитов на балансе для отправки запроса.\nКупите кредиты в разделе "Пополнить баланс".')
        return None
    file_info = await bot.get_file(message.photo[-1].file_id)
    image_url = f'https://api.telegram.org/file/bot{BOT_TOKEN}/{file_info.file_path}'
    model = settings.get('vision_model')
    try:
        response_stream = openai_client.chat.completions.create(
            model=model,
            messages=user_data['chat_history'] + [{'role': 'user', 'content': [
                {'type': 'text', 'text': message.caption},
                {'type': 'image_url', 'image_url': {'url': image_url}}
            ]}],
            temperature=settings.get('temperature'),
            stream=True
        )
        response_text, sent_message, completion_tokens = await stream_response(message, response_stream, model)
    except Exception as e:
        await message.answer('Ошибка!')
        traceback.print_exc()
        QUEUED_USERS.remove(message.from_user.id)
        return None
    
    try:
        model_pricing = await get_model_pricing(model)
    except:
        traceback.print_exc()
        QUEUED_USERS.remove(message.from_user.id)
        await message.answer('Ошибка!')
        return None
    
    prompt_tokens = sum(count_tokens(msg['content'], model) for msg in user_data['chat_history'] if msg['role'] != 'assistant')
    prompt_tokens += count_tokens(message.caption or "", model)
    spent_prompt_credits = prompt_tokens * float(model_pricing['prompt']) / 1000
    spent_completion_credits = completion_tokens * float(model_pricing['completion']) / 1000
    spent_credits = spent_prompt_credits + spent_completion_credits
    
    await db.decrease_balance(user_id, spent_credits + 1.5)  # Добавляем 1.5 кредита за обработку изображения
    await db.increase_total_chat_requests(user_id, prompt_tokens + completion_tokens)
    
    chat_history = user_data['chat_history']
    chat_history.append({'role': "user", 'content': [
        {'type': "text", 'text': message.caption},
        {'type': "image_url", 'image_url': {'url': image_url}}
    ]})
    chat_history.append({'role': "assistant", 'content': response_text})
    await db.update_user(user_id, {'chat_history': chat_history, 'balance': user_data['balance'], 'settings': settings})
    await wait.delete()
    QUEUED_USERS.remove(message.from_user.id)
    spent_credits = spent_prompt_credits + spent_completion_credits
    
    await db.decrease_balance(user_id, spent_credits)
    await db.increase_total_chat_requests(user_id, prompt_tokens + completion_tokens)
    
    await wait.delete()
    QUEUED_USERS.remove(message.from_user.id)
