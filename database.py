import aiosqlite
import orjson

SCENARIO_FORMAT = '''
You must play out a scenario called {scenario_name}.

Scenario description:
==START-SCENARIO-DESCRIPTION==
{scenario_description}
==FINISH-SCENARIO-DESCRIPTION==

Example dialog:
==START=DIALOGUE-EXAMPLES==
{dialogue_examples}
==FINISH-DIALOGUE-EXAMPLES==

Remember - you can never, ever go beyond this script!
'''

class DB:
    def __init__(self, db_path: str):
        self.db_path = db_path

    async def create_tables(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    balance REAL DEFAULT 10,
                    chat_history TEXT DEFAULT '[{"role": "system", "content": "You are a helpful assistant."}]',
                    settings TEXT DEFAULT '{"model": "openai/gpt-4o", "temperature": 0.7, "max_words": 2000, "voice": "male", "scenario": 0, "vision_model": "vis-openai/gpt-4o"}'
                )
            ''')
            await db.execute('''
                CREATE TABLE IF NOT EXISTS stats (
                    user_id INTEGER PRIMARY KEY,
                    generated_tokens INTEGER DEFAULT 0,
                    spent_credits REAL DEFAULT 0,
                    total_chat_requests INTEGER DEFAULT 0,
                    total_image_requests INTEGER DEFAULT 0,
                    total_audio_requests INTEGER DEFAULT 0,
                    total_vision_requests INTEGER DEFAULT 0
                )
            ''')
            await db.execute('''
                CREATE TABLE IF NOT EXISTS scenarios (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    scenario_name TEXT,
                    scenario_description TEXT,
                    example_dialogue TEXT
                )
            ''')
            await db.execute('''
                INSERT OR IGNORE INTO scenarios (id, user_id, scenario_name, scenario_description, example_dialogue)
                VALUES (0, 0, 'Adwizard', 'Виртуальный ассистент по имени Adwizard, который помогает пользователю.', 'Пользователь: Привет! \nAdwizard: Здравствуйте! Чем я могу вам помочь сегодня?')
            ''')
            await db.commit()

    async def get_user(self, user_id: int) -> dict:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
            row = await cursor.fetchone()
            return {
                'user_id': row[0],
                'balance': row[1],
                'chat_history': orjson.loads(row[2]),
                'settings': orjson.loads(row[3])
            } if row else None
    
    async def add_user(self, user_id: int, balance=10):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("INSERT INTO users (user_id, balance) VALUES (?, ?)",
                             (user_id, balance))
            await db.commit()

    async def get_scenarios(self, user_id: int) -> list:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute('SELECT * FROM scenarios WHERE user_id = ?', (user_id,))
            rows = await cursor.fetchall()
            return [
                {
                    'id': row[0],
                    'scenario_name': row[2],
                    'scenario_description': row[3],
                    'example_dialogue': row[4]
                }
                for row in rows
            ]
    
    async def get_scenario_by_name(self, user_id: int, scenario_name: str) -> dict:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute('SELECT * FROM scenarios WHERE user_id = ? AND scenario_name = ?', (user_id, scenario_name))
            row = await cursor.fetchone()
            return {
                'id': row[0],
                'scenario_name': row[2],
                'scenario_description': row[3],
                'example_dialogue': row[4]
            } if row else None
    
    async def add_scenario(self, user_id: int, scenario_name: str, scenario_description: str, example_dialogue: str) -> int:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute('INSERT INTO scenarios (user_id, scenario_name, scenario_description, example_dialogue) VALUES (?, ?, ?, ?)',
                                      (user_id, scenario_name, scenario_description, example_dialogue))
            await db.commit()
            return cursor.lastrowid
    
    async def get_scenario(self, scenario_id: int) -> dict:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute('SELECT * FROM scenarios WHERE id = ?', (scenario_id,))
            row = await cursor.fetchone()
            return {
                'id': row[0],
                'scenario_name': row[2],
                'scenario_description': row[3],
                'example_dialogue': row[4]
            }
    
    async def update_scenario(self, scenario_id: int, data: dict):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('UPDATE scenarios SET scenario_name = ?, scenario_description = ?, character_description = ?, example_dialogue = ? WHERE id = ?',
                             (data['scenario_name'], data['scenario_description'], data['character_description'], data['example_dialogue'], scenario_id))
            await db.commit()
    
    
    async def update_user(self, user_id: int, data: dict):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('UPDATE users SET balance = ?, chat_history = ?, settings = ? WHERE user_id = ?',
                             (data['balance'], orjson.dumps(data['chat_history']), orjson.dumps(data['settings']), user_id))
            await db.commit()
    
    async def decrease_balance(self, user_id: int, amount: float):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('UPDATE users SET balance = balance - ? WHERE user_id = ?', (amount, user_id))
            await db.execute('UPDATE stats SET spent_credits = spent_credits + ? WHERE user_id = ?', (amount, user_id))
            await db.commit()
    
    async def increase_total_chat_requests(self, user_id: int, tokens: int):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('UPDATE stats SET total_chat_requests = total_chat_requests + 1 WHERE user_id = ?', (user_id,))
            await db.execute('UPDATE stats SET generated_tokens = generated_tokens + ? WHERE user_id = ?', (tokens, user_id))
            await db.commit()

    async def increase_balance(self, user_id: int, amount: float):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('UPDATE users SET balance = balance + ? WHERE user_id = ?', (amount, user_id))
            await db.commit()
    
    async def get_settings(self, user_id: int) -> dict:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute('SELECT settings FROM users WHERE user_id = ?', (user_id,))
            row = await cursor.fetchone()
            return orjson.loads(row[0]) if row else {}
    
    async def change_model(self, user_id: int, model: str):
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute('SELECT settings FROM users WHERE user_id = ?', (user_id,))
            row = await cursor.fetchone()
            if row:
                settings = orjson.loads(row[0])
                settings['model'] = model
                await db.execute('UPDATE users SET settings = ? WHERE user_id = ?', (orjson.dumps(settings), user_id))
                await db.commit()
            else:
                raise ValueError(f"Пользователь с id {user_id} не найден")
    
    async def change_vision_model(self, user_id: int, vision_model: str):
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute('SELECT settings FROM users WHERE user_id = ?', (user_id,))
            row = await cursor.fetchone()
            if row:
                settings = orjson.loads(row[0])
                settings['vision_model'] = vision_model
                await db.execute('UPDATE users SET settings = ? WHERE user_id = ?', (orjson.dumps(settings), user_id))
                await db.commit()
    
    async def change_temperature(self, user_id: int, temperature: float):
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute('SELECT settings FROM users WHERE user_id = ?', (user_id,))
            row = await cursor.fetchone()
            if row:
                settings = orjson.loads(row[0])
                settings['temperature'] = temperature
                await db.execute('UPDATE users SET settings = ? WHERE user_id = ?', (orjson.dumps(settings), user_id))
                await db.commit()
    
    async def change_max_words(self, user_id: int, max_words: int):
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute('SELECT settings FROM users WHERE user_id = ?', (user_id,))
            row = await cursor.fetchone()
            if row:
                settings = orjson.loads(row[0])
                settings['max_words'] = max_words
                await db.execute('UPDATE users SET settings = ? WHERE user_id = ?', (orjson.dumps(settings), user_id))
                await db.commit()
            else:
                raise ValueError(f"Пользователь с id {user_id} не найден")
    
    async def change_scenario(self, user_id: int, scenario_id: int):
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute('SELECT settings FROM users WHERE user_id = ?', (user_id,))
            row = await cursor.fetchone()
            if row:
                settings = orjson.loads(row[0])
                settings['scenario'] = scenario_id
                await db.execute('UPDATE users SET settings = ? WHERE user_id = ?', (orjson.dumps(settings), user_id))
                await db.commit()
            else:
                raise ValueError(f"Пользователь с id {user_id} не найден")

    async def clear_chat(self, user_id: int):
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute('SELECT settings FROM users WHERE user_id = ?', (user_id,))
            row = await cursor.fetchone()
            if row:
                settings = orjson.loads(row[0])
                scenario_id = settings.get('scenario', 0)
                
                cursor = await db.execute('SELECT scenario_description FROM scenarios WHERE id = ?', (scenario_id,))
                scenario_row = await cursor.fetchone()
                
                if scenario_row:
                    system_message = SCENARIO_FORMAT.format(
                        scenario_name=settings.get('scenario_name', 'Adwizard'),
                        scenario_description=scenario_row[0],
                        dialogue_examples=settings.get('example_dialogue', '')
                    )
                else:
                    system_message = "You are a helpful assistant."
                
                await db.execute('UPDATE users SET chat_history = ? WHERE user_id = ?', 
                                 (orjson.dumps([{"role": "system", "content": system_message}]), user_id))
                await db.commit()
            else:
                raise ValueError(f"Пользователь с id {user_id} не найден")
    
    async def restore_balance(self, user_id: int):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('UPDATE users SET balance = 10 WHERE user_id = ?', (user_id,))
            await db.commit()