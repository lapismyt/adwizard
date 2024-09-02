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
from aiogram.types import Message, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery, LabeledPrice, PreCheckoutQuery
import os
from dotenv import load_dotenv
import orjson
import openai
import aiohttp
import traceback

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

@dp.callback_query(F.data.startswith('settings'))
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

@dp.callback_query(F.data.startswith('topup'))
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

@dp.callback_query(F.data.startswith('start'))
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
    await callback.answer(f'Модель установлена: {model_type}')

@dp.callback_query(F.data.startswith('model'))
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

@dp.callback_query(F.data.startswith('vision_model'))
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
    if model not in MODELS['recommended']['chat']:
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

@dp.callback_query(F.data.startswith('temperature'))
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

@dp.callback_query(F.data.startswith('max_words'))
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

@dp.callback_query(F.data.startswith('scenario'))
async def scenario_callback(callback: CallbackQuery, state: FSMContext):
    await callback.message.edit_text('Выберите сценарий:')
    await state.set_state(Settings.scenario)

@dp.message(StateFilter(Settings.scenario))
async def scenario_choose_callback(message: Message, state: FSMContext):
    scenario = message.text
    scenario = await db.get_scenario_by_name(message.from_user.id, scenario)
    if not scenario:
        await message.answer('Такого сценария не существует')
        await state.clear()
        return None
    await db.change_scenario(message.from_user.id, scenario['id'])
    await message.answer(f'Сценарий установлен: {scenario["scenario_name"]}\nЧтобы он начал работать, используйте /clear.')
    await state.clear()

@dp.callback_query(F.data.startswith('clear'))
async def clear_callback(callback: CallbackQuery):
    if callback.from_user.id in QUEUED_USERS:
        await callback.message.edit_text('Сначала дождитесь выполнения запроса.')
        return None
    await db.clear_chat(callback.from_user.id)
    await callback.message.edit_text('Чат очищен')

@dp.callback_query(F.data.startswith('guide'))
async def guide_callback(callback: CallbackQuery):
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text='<- Назад', callback_data='start')]
        ]
    )
    await callback.message.edit_text(f'Руководство:\n{GUIDE}', reply_markup=keyboard)

@dp.callback_query(F.data.startswith('make_scenario'))
async def make_scenario_callback(callback: CallbackQuery, state: FSMContext):
    await callback.message.edit_text('Введите название сценария:')
    await state.set_state(MakeScenario.scenario_name)

@dp.message(StateFilter(MakeScenario.scenario_name))
async def make_scenario_name_callback(message: Message, state: FSMContext):
    scenario_name = message.text
    scenario_name = await db.get_scenario_by_name(message.from_user.id, scenario_name)
    if scenario_name:
        await message.answer('Такой сценарий уже существует.\nВведите название сценария:')
        return None
    await state.update_data(scenario_name=scenario_name)
    await message.answer('Введите описание сценария:')
    await state.set_state(MakeScenario.scenario_description)

@dp.message(StateFilter(MakeScenario.scenario_description))
async def make_scenario_description_callback(message: Message, state: FSMContext):
    scenario_description = message.text
    await state.update_data(scenario_description=scenario_description)
    await message.answer('Введите примеры диалогов:')
    await state.set_state(MakeScenario.example_dialogues)

@dp.message(StateFilter(MakeScenario.example_dialogues))
async def make_scenario_example_dialogues_callback(message: Message, state: FSMContext):
    example_dialogues = message.text
    data = await state.get_data()
    scenario_name = data.get('scenario_name')
    scenario_description = data.get('scenario_description')
    scenario_id = await db.add_scenario(message.from_user.id, scenario_name, scenario_description, example_dialogues)
    await message.answer(f'Сценарий создан: {scenario_name}')
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
    return None

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
        return None
    chat_history = user_data['chat_history']
    chat_history.append({"role": "user", "content": message.text})
    try:
        response = openai_client.chat.completions.create(
            model=settings.get('model'),
            messages=chat_history,
            temperature=settings.get('temperature')
        )
    except Exception as e:
        await message.answer('Ошибка!')
        traceback.print_exc()
        return None
    chat_history.append({"role": "assistant", "content": response.choices[0].message.content})
    while sum(len(msg['content'].split()) for msg in chat_history) > settings.get('max_words'):
        if chat_history[0]['role'] == 'system':
            chat_history.pop(1)
        else:
            chat_history.pop(0)
    await db.update_user(user_id, {'chat_history': chat_history, 'balance': user_data['balance'], 'settings': settings})
    await db.increase_total_chat_requests(user_id, response.usage.total_tokens)
    model_pricing = await get_model_pricing(settings.get('model'))
    spent_credits = float(response.usage.prompt_tokens) * float(model_pricing['prompt']) + float(response.usage.completion_tokens) * float(model_pricing['completion'])
    await db.decrease_balance(user_id, float(spent_credits))
    await db.increase_total_chat_requests(user_id, response.usage.total_tokens)
    response_text = response.choices[0].message.content
    if type(response_text) is list:
        print(response_text) # debug
        response_text = response_text[0]['text']
    max_length = 4096
    if len(response_text) > max_length:
        for i in range(0, len(response_text), max_length):
            part = response_text[i:i + max_length]
            try:
                await message.answer(part, parse_mode='markdown')
            except Exception as e:
                await message.answer(part)
    else:
        try:
            await message.answer(response_text, parse_mode='markdown')
        except Exception as e:
            await message.answer(response_text)
    await wait.delete()
    QUEUED_USERS.remove(message.from_user.id)

@dp.message(F.content_type == 'photo')
async def image_callback(message: Message):
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
    try:
        response = openai_client.chat.completions.create(
            model=settings.get('vision_model'),
            messages=[{'role': 'user', 'content': [
                {'type': 'text', 'text': message.caption},
                {'type': 'image_url', 'image_url': {'url': image_url}}
            ]}],
            temperature=settings.get('temperature')
        )
    except Exception as e:
        await message.answer('Ошибка!')
        traceback.print_exc()
        return None
    model_pricing = await get_model_pricing(settings.get('vision_model'))
    spent_credits = float(response.usage.prompt_tokens) * float(model_pricing['prompt']) + float(response.usage.completion_tokens) * float(model_pricing['completion'])
    await db.decrease_balance(user_id, float(spent_credits) + 1.5)
    await db.increase_total_chat_requests(user_id, response.usage.total_tokens)
    response_text = response.choices[0].message.content
    max_length = 4096
    for i in range(0, len(response_text), max_length):
        part = response_text[i:i + max_length]
        try:
            await message.answer(part, parse_mode='markdown')
        except Exception as e:
            await message.answer(part)
    chat_history = user_data['chat_history']
    chat_history.append({'role': "user", 'content': [
        {'type': "text", 'text': message.caption},
        {'type': "image_url", 'image_url': {'url': image_url}}
    ]})
    chat_history.append({'role': "assistant", 'content': response.choices[0].message.content})
    await db.update_user(user_id, {'chat_history': chat_history, 'balance': user_data['balance'], 'settings': settings})
    await wait.delete()
    QUEUED_USERS.remove(message.from_user.id)

async def main():
    await db.create_tables()
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())