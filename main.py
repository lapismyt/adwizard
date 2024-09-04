from typing import Callable, Dict, Any, Awaitable
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
from openai import AsyncOpenAI
import aiohttp
import traceback
from base64 import b64decode
import time
import aiosqlite
from aiogram import BaseMiddleware

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

openai_client = AsyncOpenAI(
    api_key=VSEGPT_TOKEN,
    base_url=VSEGPT_URL,
    max_retries=4,
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

class AdminStates(StatesGroup):
    waiting_for_broadcast_message = State()

model_pricing_cache = {}
models_list_cache = {}

async def get_models_list():
    if 'models_list' in models_list_cache:
        return models_list_cache['models_list']
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{VSEGPT_URL.rstrip('/')}/models") as response:
            if response.status == 200:
                models_list = (await response.json())['data']
            else:
                models_list = []
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

def get_tiktoken_encoder(model: str):
    if model not in tiktoken_encoders:
        if model.startswith('openai/') or model.startswith('translate-openai/'):
            model = model.removeprefix('openai/').removeprefix('translate-openai/')
            tiktoken_encoders[model] = tiktoken.encoding_for_model(model)
        else:
            tiktoken_encoders[model] = tiktoken.get_encoding("cl100k_base")
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

async def count_images_in_chat(chat_history):
    image_count = 0
    for message in chat_history:
        if isinstance(message['content'], list):
            for content in message['content']:
                if content['type'] == 'image_url':
                    image_count += 1
    return image_count

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

@dp.message(Command('ban'))
async def ban_command(message: Message):
    if message.from_user.id == int(ADMIN_ID):
        if len(message.text.split()) < 2:
            await message.answer('Пожалуйста, укажите ID пользователя для бана.')
            return
        user_id = int(message.text.split()[1])
        await db.ban_user(user_id)
        await message.answer(f'Пользователь с ID {user_id} был забанен!')
    else:
        await message.answer('Эта команда доступна только администратору.')

@dp.message(Command('unban'))
async def unban_command(message: Message):
    if message.from_user.id == int(ADMIN_ID):
        if len(message.text.split()) < 2:
            await message.answer('Пожалуйста, укажите ID пользователя для разбана.')
            return
        user_id = int(message.text.split()[1])
        await db.unban_user(user_id)
        await message.answer(f'Пользователь с ID {user_id} был разбанен!')
    else:
        await message.answer('Эта команда доступна только администратору.')

@dp.message(Command('ban_list'))
async def ban_list_command(message: Message):
    if message.from_user.id == int(ADMIN_ID):
        await message.answer('Список забаненных пользователей:\n' + '\n'.join([str(user_id) for user_id in await db.get_banned_users()]))
    else:
        await message.answer('Эта команда доступна только администратору.')

@dp.message(Command('get_user'))
async def get_user_command(message: Message):
    if message.from_user.id == int(ADMIN_ID):
        user = await db.get_user(message.from_user.id)
        await message.answer(f'Пользователь: ```json\n{orjson.dumps(user, option=orjson.OPT_INDENT_2)}\n```', parse_mode='markdown')
    else:
        await message.answer('Эта команда доступна только администратору.')


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

@dp.message(Command('stats'))
async def stats_command(message: Message):
    # Отобразить общую статистику и статистику пользователя
    # Личная статистика: всё из таблицы stats
    # Общая статистика: количество пользователей + суммы всех значений из таблицы stats
    user_stats = await db.get_user_stats(message.from_user.id)
    total_users = await db.get_total_users()
    total_stats = await db.get_total_stats()
    await message.answer(
        f'Ваша статистика:\n'
        f'Сгенерированные токены: {user_stats["generated_tokens"]}\n'
        f'Потраченные кредиты: {user_stats["spent_credits"]}\n'
        f'Всего запросов чата: {user_stats["total_chat_requests"]}\n'
        f'Всего запросов изображений: {user_stats["total_image_requests"]}\n'
        f'Всего запросов аудио: ~{user_stats["total_audio_requests"]}~ скоро\n'
        f'Всего запросов видения: {user_stats["total_vision_requests"]}\n\n'
        f'Общая статистика:\n'
        f'Всего пользователей: {total_users}\n'
        f'Суммарные сгенерированные токены: {total_stats["generated_tokens"]}\n'
        f'Суммарные потраченные кредиты: {total_stats["spent_credits"]}\n'
        f'Суммарные запросы чата: {total_stats["total_chat_requests"]}\n'
        f'Суммарные запросы изображений: {total_stats["total_image_requests"]}\n'
        f'Суммарные запросы аудио: ~{total_stats["total_audio_requests"]}~ скоро\n'
        f'Суммарные запросы видения: {total_stats["total_vision_requests"]}',
    parse_mode='markdown')

@dp.message(Command('image'))
async def image_command(message: Message):
    if message.from_user.id in QUEUED_USERS:
        await message.answer('Сначала дождитесь выполнения предыдущего запроса.')
        return None
    QUEUED_USERS.append(message.from_user.id)
    wait = await message.answer('Подождите немного...')
    prompt = message.text.removeprefix('/image').strip()
    if not prompt:
        await message.answer('Пожалуйста, введите описание изображения.')
        return None
    user_data = await db.get_user(message.from_user.id)
    if user_data['balance'] <= 1.8 and message.from_user.id != int(ADMIN_ID):
        await message.answer('Недостаточно кредитов на балансе для отправки запроса.\nКупите кредиты в разделе "Пополнить баланс".')
        return None
    await db.decrease_balance(message.from_user.id, 1.8)
    response = await openai_client.images.generate(model=IMAGE_MODEL, prompt=prompt, n=1, size='1024x1024', response_format='b64_json')
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
        await db.update_user(message.from_user.id, {'chat_history': chat_history, 'balance': user_data['balance'], 'settings': user_data['settings']})
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
    
    # Подсчет количества изображений в чате
    image_count = await count_images_in_chat(user_data['chat_history'])
    image_cost = image_count * 1.5
    
    if user_data['balance'] < image_cost and user_id != int(ADMIN_ID):
        await message.answer(f'Недостаточно кредитов на балансе для отправки запроса.\nНеобходимо {image_cost} кредитов за изображения в чате.\nКупите кредиты в разделе "Пополнить баланс".')
        QUEUED_USERS.remove(user_id)
        return None
    
    chat_history = user_data['chat_history']
    if len(chat_history) < 2 or chat_history[-1]['role'] != 'assistant':
        await message.answer('Нет предыдущего ответа для повторной генерации.')
        QUEUED_USERS.remove(user_id)
        return None
    chat_history.pop()
    model = settings.get('model')
    try:
        response_stream = await openai_client.chat.completions.create(
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
    spent_credits = spent_prompt_credits + spent_completion_credits + image_cost
    await db.decrease_balance(user_id, spent_credits)
    await db.increase_total_chat_requests(user_id, prompt_tokens + completion_tokens)
    await db.update_user(user_id, {'chat_history': chat_history, 'balance': user_data['balance'], 'settings': settings})
    await wait.delete()
    QUEUED_USERS.remove(user_id)


@dp.callback_query(F.data == 'settings')
async def settings_callback(callback: CallbackQuery):
    settings = await db.get_settings(int(callback.from_user.id))
    scenario = await db.get_scenario(settings['scenario'])
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=f'Модель: {settings["model"]}', callback_data='model')],
            [InlineKeyboardButton(text=f'Vision-модель: {settings["vision_model"]}', callback_data='vision_model')],
            [InlineKeyboardButton(text=f'Температура: {settings["temperature"]}', callback_data='temperature')],
            [InlineKeyboardButton(text=f'Максимальное количество слов: {settings["max_words"]}', callback_data='max_words')],
            [InlineKeyboardButton(text=f'Сценарий: {scenario["scenario_name"]}', callback_data='scenario')],
            [InlineKeyboardButton(text='<- Назад', callback_data='start')]
        ]
    )
    await callback.message.edit_text('Настройки', reply_markup=keyboard)

@dp.callback_query(F.data.startswith('topup_'))
async def topup_callback(callback: CallbackQuery):
    prices = [LabeledPrice(label='Оплатить звёздами', amount=int(callback.data.split('_')[1]))]
    await bot.send_invoice(callback.from_user.id, 'Пополнение баланса', 'Покупка кредитов для использования ИИ бота Adwizard', 'invoice', 'XTR', prices)

@dp.callback_query(F.data == 'topup')
async def topup_callback(callback: CallbackQuery):
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text='Купить 10 кредитов', callback_data='topup_10')],
            [InlineKeyboardButton(text='Купить 50 кредитов', callback_data='topup_50')],
            [InlineKeyboardButton(text='Купить 100 кредитов', callback_data='topup_100')],
            [InlineKeyboardButton(text='Купить 200 кредитов', callback_data='topup_200')],
            [InlineKeyboardButton(text='Купить 500 кредитов', callback_data='topup_500')],
            [InlineKeyboardButton(text='Купить 1000 кредитов', callback_data='topup_1000')],
            [InlineKeyboardButton(text='<- Назад', callback_data='start')],
        ]
    )
    await callback.message.edit_text('Выберите количество кредитов для пополнения', reply_markup=keyboard)

@dp.pre_checkout_query()
async def pre_checkout_query(pre_checkout_query: PreCheckoutQuery):
    await bot.answer_pre_checkout_query(pre_checkout_query.id, ok=True)

@dp.message(F.content_type == 'successful_payment')
async def successful_payment(message: Message):
    amount = message.successful_payment.total_amount
    await db.increase_balance(message.from_user.id, amount)
    await message.answer(f'Платеж прошел успешно! Вы пополнили баланс на {amount} кредитов звёздами.')

@dp.callback_query(F.data == 'start')
async def start_callback(callback: CallbackQuery):
    await callback.message.edit_text(text='Главное меню', reply_markup=InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text='Очистить чат', callback_data='clear')],
            [InlineKeyboardButton(text='Настройки', callback_data='settings')],
            [InlineKeyboardButton(text='Создать сценарий', callback_data='make_scenario')],
            [InlineKeyboardButton(text='Пополнить баланс', callback_data='topup')],
            [InlineKeyboardButton(text='Руководство', callback_data='guide')],
        ]
    ))

@dp.callback_query(F.data.startswith('model_'))
async def model_callback(callback: CallbackQuery, state: FSMContext):
    if callback.data.split('_')[1] == 'choose':
        await callback.message.edit_text('Напишите название модели:')
        await state.set_state(Settings.model)
        return None
    model_type = callback.data.split('_')[1]
    model = MODELS['recommended']['chat'][model_type]
    await db.change_model(callback.from_user.id, model)
    await callback.answer(f'Модель установлена: {model}')

@dp.callback_query(F.data == 'model')
async def model_callback(callback: CallbackQuery):
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=f'Дешёвая', callback_data=f'model_cheap')],
            [InlineKeyboardButton(text=f'Умная', callback_data=f'model_best')],
            [InlineKeyboardButton(text=f'Русская', callback_data=f'model_russian')],
            [InlineKeyboardButton(text=f'Программирование', callback_data=f'model_programming')],
            [InlineKeyboardButton(text=f'Ролеплей', callback_data=f'model_roleplay')],
            [InlineKeyboardButton(text=f'Строгая', callback_data=f'model_strict')],
            [InlineKeyboardButton(text=f'Сбалансированная', callback_data=f'model_balanced')],
            [InlineKeyboardButton(text=f'Креативная', callback_data=f'model_creative')],
            [InlineKeyboardButton(text=f'Онлайн', callback_data=f'model_online')],
            [InlineKeyboardButton(text=f'Выбрать другую', callback_data=f'model_choose')],
            [InlineKeyboardButton(text=f'<- Назад', callback_data=f'settings')],
        ]
    )
    await callback.message.edit_text('Выберите модель', reply_markup=keyboard)

@dp.callback_query(F.data.startswith('vision_model_'))
async def vision_model_callback(callback: CallbackQuery):
    vision_model = callback.data.removeprefix('vision_model_')
    await db.change_vision_model(callback.from_user.id, vision_model)
    await callback.answer(f'Vision-модель установлена: {vision_model}')

@dp.callback_query(F.data == 'vision_model')
async def vision_model_callback(callback: CallbackQuery):
    vision_model_buttons = [
        [InlineKeyboardButton(text=model, callback_data=f'vision_model_{model}')] for model in MODELS['vision']
    ]
    vision_model_buttons.append([InlineKeyboardButton(text=f'<- Назад', callback_data=f'settings')])
    keyboard = InlineKeyboardMarkup(inline_keyboard=vision_model_buttons)
    await callback.message.edit_text('Выберите vision-модель', reply_markup=keyboard)

@dp.message(StateFilter(Settings.model))
async def model_choose_callback(message: Message, state: FSMContext):
    model = message.text
    if model.startswith('OMF') and not ENABLE_OMF:
        await message.answer(text='OMF модели временно отключены')
        return None
    models_list = await get_models_list()
    if (model not in MODELS['chat'] and model.removeprefix('translate-') not in MODELS['chat']) or (model not in models_list):
        await message.answer(text='Такой модели не существут\nЕсли не знаете какую модель выбрать, могу посоветовать `openai/gpt-4o-mini`', reply_markup=InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text='openai/gpt-4o-mini', callback_data='model_cheap')]
            ]
        ),
        parse_mode='markdown')
        return None
    await db.change_model(message.from_user.id, model)
    await message.answer(f'Модель установлена: {model}')
    await state.clear()

@dp.callback_query(F.data == 'temperature')
async def temperature_callback(callback: CallbackQuery, state: FSMContext):
    await callback.message.edit_text('Выберите температуру (0-2):')
    await state.set_state(Settings.temperature)

@dp.message(StateFilter(Settings.temperature))
async def temperature_choose_callback(message: Message, state: FSMContext):
    temperature = float(message.text)
    if temperature < 0 or temperature > 2:
        await message.answer('Температура должна быть от 0 до 2')
        return None
    await db.change_temperature(message.from_user.id, temperature)
    await message.answer(f'Температура установлена: {temperature}')
    await state.clear()

@dp.callback_query(F.data == 'max_words')
async def max_words_callback(callback: CallbackQuery, state: FSMContext):
    await callback.message.edit_text('Выберите максимальное количество слов (10-10000):')
    await state.set_state(Settings.max_words)

@dp.message(StateFilter(Settings.max_words))
async def max_words_choose_callback(message: Message, state: FSMContext):
    max_words = int(message.text)
    if max_words < 10 or max_words > 10000:
        await message.answer('Максимальное количество слов должно быть от 10 до 10000')
        return None
    await db.change_max_words(message.from_user.id, max_words)
    await message.answer(f'Максимальное количество слов установлено: {max_words}')
    await state.clear()

@dp.callback_query(F.data == 'scenario')
async def scenario_callback(callback: CallbackQuery, state: FSMContext):
    await callback.message.edit_text('Выберите сценарий:')
    await state.set_state(Settings.scenario)

@dp.message(StateFilter(Settings.scenario))
async def scenario_choose_callback(message: Message, state: FSMContext):
    scenario = message.text
    scenario = await db.get_scenario_by_name(scenario)
    if not scenario:
        await message.answer('Такого сценария не существует')
        await state.clear()
        return None
    await db.change_scenario(message.from_user.id, scenario['id'])
    await message.answer(f'Сценарий установлен: {scenario["scenario_name"]}\nЧтобы он начал работать, исползуйте /clear.')
    await state.clear()

@dp.callback_query(F.data == 'clear')
async def clear_callback(callback: CallbackQuery):
    if callback.from_user.id in QUEUED_USERS:
        await callback.answer('Сначала дождитесь выполнения запроса.')
        return None
    await db.clear_chat(callback.from_user.id)
    await callback.answer('Чат очищен')

@dp.callback_query(F.data == 'guide')
async def guide_callback(callback: CallbackQuery):
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text='<- Назад', callback_data='start')]
        ]
    )
    await callback.message.edit_text(f'Руководство:\n{GUIDE}', reply_markup=keyboard)

@dp.callback_query(F.data == 'make_scenario')
async def make_scenario_callback(callback: CallbackQuery, state: FSMContext):
    await callback.message.edit_text('Введите название сценария:')
    await state.set_state(MakeScenario.scenario_name)

@dp.message(StateFilter(MakeScenario.scenario_name))
async def make_scenario_name_callback(message: Message, state: FSMContext):
    scenario_name = message.text
    existing_scenario = await db.get_scenario_by_name(scenario_name)
    if existing_scenario:
        await message.answer('Такой сценарий уже существует.\nВведите другое название сценария:')
        return None
    await state.update_data(scenario_name=scenario_name)
    await message.answer(f'Название сценария: {scenario_name}\nТеперь введите описание сценария:')
    await state.set_state(MakeScenario.scenario_description)

@dp.message(StateFilter(MakeScenario.scenario_description))
async def make_scenario_description_callback(message: Message, state: FSMContext):
    state_data = await state.get_data()
    scenario_description = state_data.get('scenario_description', '') + '\n' + message.text
    await state.update_data(scenario_description=scenario_description)
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text='Готово', callback_data='scenario_descr_done')]
        ]
    )
    await message.answer('Продолжите описание или нажмите "Готово":', reply_markup=keyboard)

@dp.callback_query(F.data == 'scenario_descr_done')
async def make_scenario_example_dialogues_callback(callback: CallbackQuery, state: FSMContext):
    state_data = await state.get_data()
    await callback.message.edit_text(f'Название: {state_data.get("scenario_name")}\nОписание: {state_data.get("scenario_description")[:500]}...\n\nТеперь введите примеры диалогов:')
    await state.set_state(MakeScenario.example_dialogues)

@dp.message(StateFilter(MakeScenario.example_dialogues))
async def make_scenario_example_dialogues_callback(message: Message, state: FSMContext):
    state_data = await state.get_data()
    example_dialogues = state_data.get('example_dialogues', '') + '\n\n' + message.text
    await state.update_data(example_dialogues=example_dialogues)
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text='Готово', callback_data='scenario_dialogues_done')]
        ]
    )
    await message.answer('Продолжите примеры диалогов или нажмите "Готово":', reply_markup=keyboard)

@dp.callback_query(F.data == 'scenario_dialogues_done')
async def make_scenario_example_dialogues_callback(callback: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    scenario_name = data.get('scenario_name')
    scenario_description = data.get('scenario_description')
    example_dialogues = data.get('example_dialogues')
    if not scenario_name or not scenario_description or not example_dialogues:
        await callback.message.answer('Ошибка: не все данные сценария заполнены. Пожалуйста, начните создание сценария заново.')
        await state.clear()
        return
    scenario_id = await db.add_scenario(callback.from_user.id, scenario_name, scenario_description, example_dialogues)
    await callback.message.answer(f'Сценарий создан:\nНазвание: {scenario_name}\nОписание: {scenario_description[:500]}...\nПримеры диалогов: {example_dialogues[:100]}...')
    await state.clear()

@dp.message(Command('restore_balance'))
async def restore_balance_command(message: Message):
    if message.from_user.id != int(ADMIN_ID):
        await message.answer('У вас нет доступа к этой команде')
        return None
    await db.restore_balance(message.from_user.id)
    await message.answer('Баланс восстановлен')

async def get_models_list():
    async with aiohttp.ClientSession() as session:
        async with session.get(VSEGPT_URL.rstrip('/') + '/models') as response:
            if response.status == 200:
                return (await response.json())['data']
            else:
                return []

async def get_model_pricing(model_name):
    models_list = await get_models_list()
    for model in models_list:
        if model['id'] == model_name:
            return model['pricing']
    return {
        'prompt': 0.1,
        'completion': 0.15
    }

@dp.message(Command('broadcast'))
async def cmd_broadcast(message: Message, state: FSMContext):
    if str(message.from_user.id) != ADMIN_ID:
        await message.answer('У вас нет прав для использования этой команды.')
        return
    await message.answer('Введите сообщение для рассылки:')
    await state.set_state(AdminStates.waiting_for_broadcast_message)

@dp.message(StateFilter(AdminStates.waiting_for_broadcast_message))
async def process_broadcast_message(message: Message, state: FSMContext):
    if str(message.from_user.id) != ADMIN_ID:
        await message.answer('У вас нет прав для использования этой команды.')
        await state.clear()
        return
    broadcast_message = message.text
    users = await db.get_all_users()
    sent_count = 0
    for user in users:
        try:
            await bot.send_message(user['id'], broadcast_message)
            sent_count += 1
        except Exception as e:
            print(f'Failed to send message to user {user["id"]}: {e}')
    await message.answer(f'Рассылка заверше��а. Отправлено {sent_count} из {len(users)} пользователей.')
    await state.clear()

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
    
    # Подсчет количества изображений в чате
    image_count = await count_images_in_chat(user_data['chat_history'])
    image_cost = image_count * 1.5
    
    if user_data['balance'] < image_cost and user_id != int(ADMIN_ID):
        await message.answer(f'Недостаточно кредитов на балансе для отправки запроса.\nНеобходимо {image_cost} кредитов за изображения в чате.\nКупите кредиты в разделе "Пополнить баланс".')
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
        
        response_stream = await openai_client.chat.completions.create(
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
    spent_credits = spent_prompt_credits + spent_completion_credits + image_cost
    
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
    
    # Подсчет количества изображений в чате + новое изображение
    image_count = await count_images_in_chat(user_data['chat_history']) + 1
    image_cost = image_count * 1.5
    
    if user_data['balance'] < image_cost and user_id != int(ADMIN_ID):
        await message.answer(f'Недостаточно кредитов на балансе для отправки запроса.\nНеобходимо {image_cost} кредитов за изображения в чате.\nКупите кредиты в разделе "Пополнить баланс".')
        QUEUED_USERS.remove(message.from_user.id)
        return None
    
    file_info = await bot.get_file(message.photo[-1].file_id)
    image_url = f'https://api.telegram.org/file/bot{BOT_TOKEN}/{file_info.file_path}'
    model = settings.get('vision_model')
    try:
        response_stream = await openai_client.chat.completions.create(
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
    spent_credits = spent_prompt_credits + spent_completion_credits + image_cost
    await db.decrease_balance(user_id, spent_credits)
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
    
    # Подсчет количества изображений в чате
    image_count = await count_images_in_chat(user_data['chat_history'])
    image_cost = image_count * 1.5
    
    if user_data['balance'] < image_cost and user_id != int(ADMIN_ID):
        await message.answer(f'Недостаточно кредитов на балансе для отправки запроса.\nНеобходимо {image_cost} кредитов за изображения в чате.\nКупите кредиты в разделе "Пополнить баланс".')
        QUEUED_USERS.remove(user_id)
        return None
    
    chat_history = user_data['chat_history']
    if len(chat_history) < 2 or chat_history[-1]['role'] != 'assistant':
        await message.answer('Нет предыдущего ответа для повторной генерации.')
        QUEUED_USERS.remove(user_id)
        return None
    chat_history.pop(-1)
    last_message = chat_history[-1]
    if isinstance(last_message['content'], list):
        for content in last_message['content']:
            if content['type'] == 'image_url':
                image_url = content['image_url']['url']
                break
        else:
            image_url = None
    else:
        image_url = None
    try:
        if image_url:
            model = settings.get('vision_model')
            response_stream = await openai_client.chat.completions.create(
                model=model,
                messages=chat_history,
                temperature=settings.get('temperature'),
                stream=True
            )
        else:
            model = settings.get('model')
            response_stream = await openai_client.chat.completions.create(
                model=model,
                messages=chat_history,
                temperature=settings.get('temperature'),
                stream=True
            )
        response_text, sent_message, completion_tokens = await stream_response(message, response_stream, model)
    except Exception as e:
        await message.answer('Ошибка!')
        traceback.print_exc()
        QUEUED_USERS.remove(user_id)
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
    try:
        model_pricing = await get_model_pricing(model)
    except:
        traceback.print_exc()
        QUEUED_USERS.remove(user_id)
        await message.answer('Ошибка!')
        return None
    
    prompt_tokens = sum(count_tokens(msg['content'], model) for msg in chat_history if msg['role'] != 'assistant')
    spent_prompt_credits = prompt_tokens * float(model_pricing['prompt']) / 1000
    spent_completion_credits = completion_tokens * float(model_pricing['completion']) / 1000
    spent_credits = spent_prompt_credits + spent_completion_credits + image_cost
    
    await db.decrease_balance(user_id, spent_credits)
    await db.increase_total_chat_requests(user_id, prompt_tokens + completion_tokens)
    await db.update_user(user_id, {'chat_history': chat_history, 'balance': user_data['balance'], 'settings': settings})
    await wait.delete()
    QUEUED_USERS.remove(user_id)

class UserMiddleware(BaseMiddleware):
    async def __call__(
        self,
        handler: Callable[[Message, Dict[str, Any]], Awaitable[Any]],
        event: Message | CallbackQuery,
        data: Dict[str, Any]
    ) -> Any:
        user_id = event.from_user.id
        user = await db.get_user(user_id)
        
        if not user:
            await db.add_user(user_id, balance=TEST_BALANCE)
            user = await db.get_user(user_id)
        
        if user['is_banned']:
            if isinstance(event, Message):
                await event.answer("Вы заблокированы и не можете использовать бота.")
            elif isinstance(event, CallbackQuery):
                await event.answer("Вы заблокированы и не можете использовать бота.", show_alert=True)
            return
        
        data['user'] = user
        return await handler(event, data)

async def main():
    dp.message.middleware(UserMiddleware())
    dp.callback_query.middleware(UserMiddleware())
    await db.create_tables()
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())
