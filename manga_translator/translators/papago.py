
# -*- coding: utf-8 -*-
import asyncio
import requests

from .common import CommonTranslator, InvalidServerResponse

class PapagoTranslator(CommonTranslator):
    _LANGUAGE_CODE_MAP = {
        'CHS': 'zh-CN',
        'CHT': 'zh-TW',
        'JPN': 'ja',
        'ENG': 'en',
        'KOR': 'ko',
        'VIN': 'vi',
        'FRA': 'fr',
        'DEU': 'de',
        'ITA': 'it',
        'PTB': 'pt',
        'RUS': 'ru',
        'ESP': 'es',
        'THA': 'th',
        'IND': 'id'
    }
    _HOME_URL = 'https://papago.naver.com/'
    _API_URL = 'https://papago.naver.com/api/text/translation'

    async def _translate(self, from_lang, to_lang, queries):
        if not queries:
            return []
        data = {}
        data['honorific'] = "false"
        data['source'] = from_lang
        data['target'] = to_lang
        data['text'] = '\n'.join(queries)
        data['dict'] = 'false'
        data['useGlossary'] = 'false'
        result = await self._do_request(data)
        if not isinstance(result, dict) or not isinstance(result.get('translatedText'), str):
            raise InvalidServerResponse('Papago response is missing translatedText; the web API may have changed.')
        result_list = [str.strip() for str in result["translatedText"].split("\n")]
        return result_list

    async def _do_request(self, data):
        # requests also honors Windows system proxies. Run it off the event loop.
        return await asyncio.to_thread(self._request_sync, data)

    def _request_sync(self, data):
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": self._HOME_URL,
            "Origin": self._HOME_URL.rstrip('/'),
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        }
        with requests.Session() as session:
            session.headers.update(headers)
            # Establish the same session cookies as the current web client.
            with session.get(self._HOME_URL, timeout=30) as resp:
                if resp.status_code != 200:
                    raise InvalidServerResponse(f'Papago homepage returned HTTP {resp.status_code}.')
            with session.post(self._API_URL, data=data, timeout=30) as resp:
                if resp.status_code != 200:
                    raise InvalidServerResponse(f'Papago translation returned HTTP {resp.status_code}.')
                try:
                    return resp.json()
                except ValueError as exc:
                    raise InvalidServerResponse('Papago returned a non-JSON response.') from exc
