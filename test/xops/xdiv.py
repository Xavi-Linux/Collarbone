
def xd(a:int, b:int) -> float:
    try:
        result = a/b
        return result

    except ZeroDivisionError:
        return 0
