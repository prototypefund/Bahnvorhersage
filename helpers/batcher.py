def batcher(list_of_things: list, batch_size=100):
    for i in range(0, len(list_of_things), batch_size):
        yield list_of_things[i : i + batch_size]
