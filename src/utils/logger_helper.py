import logging

def log_title (logger : logging.Logger, title, width = 60):
                line = f"{title}"
                logger.info(line.center(width, "="))