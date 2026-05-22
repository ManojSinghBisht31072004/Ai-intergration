class InputTooLongError(Exception):
    pass

class OutputValidationError(Exception):
    pass

class GatewayTimeoutError(Exception):
    pass

class RetryExhaustedError(Exception):
    pass