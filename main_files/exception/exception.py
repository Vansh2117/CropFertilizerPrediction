import sys

class CropFertilizerException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)

        # Default values
        self.lineno = None
        self.filename = None

        try:
            _, _, exc_tb = error_detail.exc_info()
            if exc_tb is not None:
                self.lineno = exc_tb.tb_lineno
                self.filename = exc_tb.tb_frame.f_code.co_filename
        except:
            pass

    def __str__(self):
        return f"Error in file [{self.filename}] at line [{self.lineno}]: {super().__str__()}"
