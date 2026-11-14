from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from manga_translator.translators.papago import PapagoTranslator
from manga_translator.translators.common import InvalidServerResponse


@pytest.mark.asyncio
async def test_translation_preserves_lines():
    translator = PapagoTranslator()
    translator._do_request = AsyncMock(return_value={'translatedText': ' 你好。\n 去哪里？ '})
    assert await translator._translate('ko', 'zh-CN', ['안녕하세요.', '어디로 가요?']) == ['你好。', '去哪里？']
    data = translator._do_request.call_args.args[0]
    assert data['source'] == 'ko'
    assert data['target'] == 'zh-CN'
    assert data['text'] == '안녕하세요.\n어디로 가요?'


@pytest.mark.asyncio
@pytest.mark.parametrize('response', [{'errorCode': 'failed'}, [], {'translatedText': None}])
async def test_invalid_response(response):
    translator = PapagoTranslator()
    translator._do_request = AsyncMock(return_value=response)
    with pytest.raises(InvalidServerResponse):
        await translator._translate('en', 'zh-CN', ['Hello'])


@pytest.mark.asyncio
async def test_empty_input_does_not_request():
    translator = PapagoTranslator()
    translator._do_request = AsyncMock()
    assert await translator._translate('en', 'zh-CN', []) == []
    translator._do_request.assert_not_awaited()


def test_web_session_uses_new_endpoint():
    with patch('manga_translator.translators.papago.requests.Session') as factory:
        session = factory.return_value.__enter__.return_value
        home = MagicMock(status_code=200)
        response = MagicMock(status_code=200)
        response.json.return_value = {'translatedText': '你好'}
        session.get.return_value.__enter__.return_value = home
        session.post.return_value.__enter__.return_value = response
        translator = PapagoTranslator()
        assert translator._request_sync({'text': 'Hello'}) == {'translatedText': '你好'}
        assert session.post.call_args.args[0] == 'https://papago.naver.com/api/text/translation'


def test_http_error_is_reported():
    with patch('manga_translator.translators.papago.requests.Session') as factory:
        session = factory.return_value.__enter__.return_value
        session.get.return_value.__enter__.return_value.status_code = 403
        with pytest.raises(InvalidServerResponse, match='HTTP 403'):
            PapagoTranslator()._request_sync({'text': 'Hello'})
        session.post.assert_not_called()
