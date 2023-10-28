import discord
from discord.ext import tasks, commands
from config import discord_bot_token
import datetime
import requests
from database.unique_change import UniqueChange
from database.plan_by_id_v2 import PlanByIdV2
from database.engine import sessionfactory
from typing import Callable, Coroutine, Union
import traceback
import functools
import asyncio


def to_thread(func: Callable) -> Coroutine:
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        wrapped = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(None, wrapped)
    return wrapper


@to_thread
def monitor_plan() -> Union[str, None]:
    global old_plan_count

    date_to_check = datetime.datetime.combine(
        datetime.date.today(),
        (datetime.datetime.now() + datetime.timedelta(hours=3)).time()
    )

    # Plan (hourly)
    message = None
    try:
        with Session() as session:
            new_plan_count = PlanByIdV2.count_entries(session)
        plan_count_delta = new_plan_count - old_plan_count
        old_plan_count = new_plan_count
        if plan_count_delta < 500:
            message = '@everyone The plan gatherer is not working, as {} new entries where added to database at {}'\
                    .format(str(plan_count_delta), str(date_to_check))
        print('checked plan ' + str(date_to_check) + ': ' + str(plan_count_delta) + ' rows were added')
    except FileExistsError as e:
        message = '@everyone Error reading Database:\n{}' \
            .format(''.join(traceback.format_exception(e, e, e.__traceback__)))
        print(message)
        print('checked ' + str(date_to_check) + ': ???? rows were added')
    
    return message


@to_thread
def monitor_change() -> Union[str, None]:
    global old_change_count

    date_to_check = datetime.datetime.combine(
        datetime.date.today(),
        (datetime.datetime.now() + datetime.timedelta(hours=3)).time()
    )

    # Recent changed (crawled every two minutes but only checked once an hour)
    message = None
    try:
        with Session() as session:
            new_change_count = UniqueChange.count_entries(session)
        count_delta = new_change_count - old_change_count
        old_change_count = new_change_count
        if count_delta < 1000:
            message = '''@everyone The recent change gatherer is not working, as {} 
                    new entries where added to database at {}'''\
                    .format(str(count_delta), str(date_to_check))
        old_change_count = new_change_count
        print('checked changes ' + str(date_to_check) + ': ' + str(count_delta) + ' rows were added')
    except Exception as e:
        message = '@everyone Error reading Database:\n{}' \
            .format(''.join(traceback.format_exception(None, e, e.__traceback__)))
        print(message)
        print('checked ' + str(date_to_check) + ': ???? rows were added')

    return message


async def monitor_website():
    channel = client.get_channel(720671295129518232)
    try:
        print('testing https://bahnvorhersage.de...')
        page = requests.get('https://bahnvorhersage.de')
        if page.ok:
            print('ok')
        else:
            message = str(datetime.datetime.now()) \
                + ': @everyone Something not working on the website:\n{}'.format(str(page))
            print(message)
            await channel.send(message)
    except Exception as ex:
        message = str(datetime.datetime.now()) + ': @everyone Error on the website:\n{}'.format(str(ex))
        print(message)
        await channel.send(message)

    try:
        print('testing https://bahnvorhersage.de/api/trip from Tübingen Hbf to Köln Hbf...')
        search = {
            'start': 'Tübingen Hbf',
            'destination': 'Köln Hbf',
            'date': (datetime.datetime.now() + datetime.timedelta(hours=1)).strftime('%d.%m.%Y %H:%M'),
            'search_for_arrival': False
        }
        trip = requests.post('https://bahnvorhersage.de/api/trip', json=search)
        if trip.ok:
            print('ok')
        else:
            message = str(datetime.datetime.now()) \
                + ': @everyone Something not working on the website:\n{}'.format(str(trip))
            print(message)
            await channel.send(message)
    except Exception as e:
        message = str(datetime.datetime.now()) + ': @everyone Something not working on the website:\n{}' \
            .format(''.join(traceback.format_exception(None, e, e.__traceback__)))
        await channel.send(message)
        print(message)


class Monitor(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.old_change_count = 0

    async def setup_hook(self) -> None:
        self.monitor.start()

    async def on_ready(self):
        print(f'{client.user} has connected to Discord!')
        channel = client.get_channel(720671295129518232)
        await channel.send('Data gatherer v2 monitor now active')

    @tasks.loop(hours=1)
    async def monitor(self):
        await client.wait_until_ready()
        channel = client.get_channel(720671295129518232)

        message = await monitor_plan()
        if message is not None:
            await channel.send(message)

        message = await monitor_change()
        if message is not None:
            await channel.send(message)

        await monitor_website()


if __name__ == "__main__":
    import helpers.bahn_vorhersage

    engine, Session = sessionfactory()

    intents = discord.Intents.default()
    intents.message_content = True

    client = Monitor(intents=intents)
    # client = discord.Client(intents=intents)

    old_change_count = 0
    old_plan_count = 0

    client.run(discord_bot_token)
