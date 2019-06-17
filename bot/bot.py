import re

from telegram.ext import (Updater, ConversationHandler, CommandHandler, MessageHandler, RegexHandler, Filters)
import logging

from bot.utils import (clean_files, save_variable, get_variable_names, read_variables, get_image_from_variable,
                       convert_image_to_io)


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
VAR_NAME, VAR_VALUE = range(2)


def read_token():
    with open("token") as f:
        res = f.readline()
    return res


def get_args():
    return {
        "proxy_url": "http://54.39.24.33:3128/"
    }


def start(update, context):
    context.bot.send_message(chat_id=update.message.chat_id, text="It's a face calculator!")
    logger.info("Started conversation with user {}".format(update.message.chat_id))

    return VAR_NAME


def variable_name(update, context):
    if len(context.args) == 1 and re.fullmatch("[a-zA-Z_]+", context.args[0]):
        context.user_data["var_name"] = context.args[0]
        logger.info("User {} set variable: {}".format(update.message.chat_id, context.user_data["var_name"]))
        return VAR_VALUE
    else:
        context.bot.send_message(chat_id=update.message.chat_id, text="Variable name should contain only letters and _")
        logger.info("User {} failed in setting variable".format(update.message.chat_id))
        return ConversationHandler.END


def variable_value(update, context):
    photo_file = update.message.photo[-1].get_file()
    save_variable(update.message.chat_id, context.user_data["var_name"], photo_file)
    logger.info("User {} assigned value to variable: {}".format(update.message.chat_id,
                                                                context.user_data["var_name"])
                )
    return ConversationHandler.END


def cancel(update, context):
    logger.info("User {} canceled assigning".format(update.message.chat_id))
    return ConversationHandler.END


def evaluate(update, context):
    if len(context.args) == 0:
        context.bot.send_message(chat_id=update.message.chat_id, text="You should pass any expression")
        logger.info("User {} didn't pass expression".format(update.message.chat_id))
    expression = "".join(context.args)
    try:
        names = get_variable_names(expression)
    except Exception as e:
        logger.info("Failed reading user {} variables names from expression {}".format(
            update.message.chat_id, expression
        ))
        logger.info(e)
        return
    try:
        variables = read_variables(update.message.chat_id, names)
    except Exception as e:
        logger.info("Failed reading user {} variables from names {}".format(
            update.message.chat_id, names
        ))
        logger.info(e)
        return
    try:
        result_variable = eval(expression, {}, variables)
    except Exception as e:
        logger.info("Failed to evaluate user {} expression {} with variables".format(
            update.message.chat_id, expression, variables
        ))
        logger.info(e)
        return

    image = get_image_from_variable(result_variable)
    image_bio = convert_image_to_io(image)
    context.bot.send_photo(chat_id=update.message.chat_id, photo=image_bio)


def main(token=None, args=None):
    if not token:
        token = read_token()
    if not args:
        args = get_args()

    updater = Updater(token=token, use_context=True, request_kwargs=args)
    dispatcher = updater.dispatcher

    start_handler = CommandHandler("start", start)
    var_name_handler = CommandHandler("set", variable_name, pass_user_data=True)
    var_value_handler = MessageHandler(Filters.photo, variable_value, pass_user_data=True)

    conversation_handler = ConversationHandler(
        entry_points=[start_handler, var_name_handler],
        states={
            VAR_NAME: [var_name_handler],
            VAR_VALUE: [var_value_handler]
        },
        fallbacks=[CommandHandler('cancel', cancel)]
    )
    dispatcher.add_handler(conversation_handler)

    eval_handler = CommandHandler("eval", evaluate)
    dispatcher.add_handler(eval_handler)

    updater.start_polling()

