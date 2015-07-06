import requests


class MyriaError(Exception):
    def __init__(self, err=None):
        if isinstance(err, requests.Response):
            msg = 'Error {} ({})'.format(err.status_code, err.reason)
            if err.text:
                msg = '{}: {}'.format(msg, err.text)
            Exception.__init__(self, msg)
        else:
            Exception.__init__(self, err)
