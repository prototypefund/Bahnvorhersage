import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
)


if __name__ == '__main__':
    logging.debug('Hello World')
