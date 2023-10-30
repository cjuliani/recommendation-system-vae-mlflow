def handle_bool_parsing(input_value, input_name):
    if input_value in ['True', 'False']:
        return input_value
    else:
        raise Exception(f"Input value '{input_name}' must be either True or False.")
