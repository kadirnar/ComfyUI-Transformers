class IntToString:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "convert"
    CATEGORY = "Transformers/Utility"

    def convert(self, value):
        return (str(value),)


class StringToInt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "0"}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "convert"
    CATEGORY = "Transformers/Utility"

    def convert(self, text):
        return (int(text.strip()),)


class FloatToString:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0.0}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "convert"
    CATEGORY = "Transformers/Utility"

    def convert(self, value):
        return (str(value),)


class StringToFloat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "0.0"}),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "convert"
    CATEGORY = "Transformers/Utility"

    def convert(self, text):
        return (float(text.strip()),)


NODE_CLASS_MAPPINGS = {
    "IntToString": IntToString,
    "StringToInt": StringToInt,
    "FloatToString": FloatToString,
    "StringToFloat": StringToFloat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IntToString": "Int to String",
    "StringToInt": "String to Int",
    "FloatToString": "Float to String",
    "StringToFloat": "String to Float",
}
