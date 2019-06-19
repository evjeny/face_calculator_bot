import re

from telegram.ext import (Updater, ConversationHandler, CommandHandler, MessageHandler,Filters)
import logging

from bot.utils import (clean_all_files, save_variable, get_variable_names, read_variables, get_image_from_variable,
                       convert_image_to_io, add_file_to_queue, remove_from_queue, RepeatedTimer, handle_message_edit)


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
VAR_NAME, VAR_VALUE = range(2)

with open("bot/help_message.txt") as help_file:
    help_message = help_file.read()


def read_token():
    with open("token") as f:
        res = f.readline()
    return res


def get_args():
    with open("proxy") as f:
        proxy = f.readline()
    return {
        "proxy_url": proxy
    }


@handle_message_edit
def start_function(update, context):
    context.bot.send_message(chat_id=update.message.chat_id, text="It's a face calculator!\n" +
                                                                  "Print /help to get list of commands")
    logger.info("Started conversation with user {}".format(update.message.chat_id))

    return VAR_NAME


@handle_message_edit
def help_function(update, context):
    context.bot.send_message(chat_id=update.message.chat_id, text=help_message)
    logger.info("User {} requested help".format(update.message.chat_id))


@handle_message_edit
def variable_name_function(update, context):
    if len(context.args) == 1 and re.fullmatch("[a-zA-Z_]+", context.args[0]):
        context.user_data["var_name"] = context.args[0]
        logger.info("User {} set variable: {}".format(update.message.chat_id, context.user_data["var_name"]))
        return VAR_VALUE
    else:
        context.bot.send_message(chat_id=update.message.chat_id, text="Variable name should contain only letters and _")
        logger.info("User {} failed in setting variable".format(update.message.chat_id))
        return ConversationHandler.END


@handle_message_edit
def variable_value_function(update, context):
    photo_file = update.message.photo[-1].get_file()
    save_variable(update.message.chat_id, context.user_data["var_name"], photo_file)
    add_file_to_queue(update.message.chat_id, context.user_data["var_name"])
    logger.info("User {} assigned value to variable: {}".format(update.message.chat_id,
                                                                context.user_data["var_name"])
                )
    return ConversationHandler.END


@handle_message_edit
def cancel_function(update, context):
    logger.info("User {} canceled assigning".format(update.message.chat_id))
    return ConversationHandler.END


@handle_message_edit
def evaluate_function(update, context):
    chat_id = update.message.chat_id

    if len(context.args) == 0:
        context.bot.send_message(chat_id=chat_id, text="You should pass any expression")
        logger.info("User {} didn't pass expression".format(chat_id))
    expression = "".join(context.args)
    try:
        names = get_variable_names(expression)

        vulnerabilities = [not re.fullmatch("[a-zA-Z_]+", name) for name in names]
        pwn = True in vulnerabilities
        if pwn:
            logger.info("User {} tried to pwn your file system with names: {}".format(
                chat_id, [names[i] for i in range(len(names)) if vulnerabilities[i]]
            ))
            raise Exception("Unacceptable variable name")

        variables = read_variables(chat_id, names)
        result_variable = eval(expression, {}, variables)
    except Exception as e:
        logger.info("Failed reading user {} variables names from expression {}".format(chat_id, expression))
        logger.info(e)
    else:
        image = get_image_from_variable(result_variable)
        image_bio = convert_image_to_io(image)
        context.bot.send_photo(chat_id=chat_id, photo=image_bio)


def main(token=None, args=None):
    if not token:
        token = read_token()
    if not args:
        args = get_args()

    # init bot
    updater = Updater(token=token, use_context=True, request_kwargs=args)
    dispatcher = updater.dispatcher

    # clean files from last sessions
    clean_all_files()

    # adding handlers for commands
    help_handler = CommandHandler("help", help_function)
    dispatcher.add_handler(help_handler)

    start_handler = CommandHandler("start", start_function)
    var_name_handler = CommandHandler("set", variable_name_function, pass_user_data=True)
    var_value_handler = MessageHandler(Filters.photo, variable_value_function, pass_user_data=True)

    conversation_handler = ConversationHandler(
        entry_points=[start_handler, help_handler, var_name_handler, var_value_handler],
        states={
            VAR_NAME: [var_name_handler],
            VAR_VALUE: [var_value_handler]
        },
        fallbacks=[CommandHandler('cancel', cancel_function)]
    )
    dispatcher.add_handler(conversation_handler)
    dispatcher.add_handler(start_handler)

    eval_handler = CommandHandler("eval", evaluate_function)
    dispatcher.add_handler(eval_handler)

    # periodically clean files from storage
    check_interval = 60 # check files every minute
    files_lifetime = 60 * 30 # file stays in filesystem only for 30 minutes
    timer = RepeatedTimer(check_interval, remove_from_queue, files_lifetime)

    # start bot
    updater.start_polling()

