import ast

class ParamConverter:
    def __init__(self):
        pass

    def _convert_param(self, v):
        if isinstance(v, str):
            v = v.strip("'\"")  # Remove surrounding quotes

            # Handle None, booleans
            if v.lower() == "none":
                return None
            elif v.lower() == "true":
                return True
            elif v.lower() == "false":
                return False

            # Try parsing numbers directly
            try:
                if "." in v or "e" in v.lower():  # Likely a float or scientific notation
                    return float(v)
                return int(v)  # Try integer conversion
            except ValueError:
                pass  # Continue processing

            # Use `literal_eval` for safer conversion of lists, tuples, and dictionaries
            try:
                parsed = ast.literal_eval(v)
                if isinstance(parsed, (list, tuple, dict)):
                    return parsed
            except (ValueError, SyntaxError):
                pass  # Ignore errors and treat as a string

            return v  # Return as a string if no conversion succeeded

        return v
