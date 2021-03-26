from typing import List, Type


def get_class_from_name(class_name: str, classes: List, err_msg: str) -> Type:
    """
    Return class from class name.

    :param class_name: String with class name.
    :param classes: List of classes to match the given class_name.
    :param err_msg: Helper string for error message.
    :return: Class.
    """
    for s in classes:
        if class_name == s.__name__:
            return s
    raise Exception(f"{err_msg} class name is not valid: {class_name}.")
