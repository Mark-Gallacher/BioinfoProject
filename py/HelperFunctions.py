def check_type(arg_value, arg_name: str, exp_type) -> None:
    """Function to check the argument value (`arg_value`) is the expected type (`exp_type`).
    Generates an Error message and raises a TypeError.
    """
    ## check if we expect one type or multiple
    if isinstance(exp_type, list):

        ## check if the type is one of the expected ones
        if not any(isinstance(arg_value, e) for e in exp_type):

            raise TypeError(
                f"""Error with `{arg_name}` not the expected type:
                  - Expected: {[e.__name__ + " , " for e in exp_type]} 
                  - Received: {arg_value} ({type(arg_value).__name__})"""
            )

        ## If arg is one of the expected types
        return None

    ## if arg is expected to be just one type
    if not isinstance(arg_value, exp_type):
        raise TypeError(
            f"""Error with `{arg_name}` not the expected type:
              - Expected: {exp_type.__name__} 
              - Received: {arg_value} ({type(arg_value).__name__})"""
        )


def check_types(args: list) -> None:
    """Function to iterate over a tuple containing (value, name, type) for an argument. Input validation for a function - see HelperFunctions.check_type() for order."""
    for arg_value, arg_name, exp_type in args:

        check_type(arg_value, arg_name, exp_type)


# check_types(("a", "b", [str, int]))
