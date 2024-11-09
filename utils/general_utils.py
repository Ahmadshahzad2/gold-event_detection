import json

def load_json(file_path):
    """
    Load and return the contents of a JSON file.

    Parameters:
    - file_path (str): The path to the JSON file.

    Returns:
    - dict: The contents of the JSON file as a Python dictionary.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print("The file was not found. Please check the file path.")
        return None
    except json.JSONDecodeError:
        print("The file is not a valid JSON. Please check the JSON syntax.")
        return None