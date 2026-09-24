"""Offline, synthetic-context checks of the real chat route; not user testing."""
import numpy as np
import pandas as pd
import pytest
import time
from fastapi.testclient import TestClient
import api


@pytest.fixture
def advisor(monkeypatch, tmp_path):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    monkeypatch.setattr(api, 'fetch_live_market_prices', lambda: {})
    path = tmp_path/'market.csv'
    pd.DataFrame(dict(close=[50000.]*50, rsi_14=[80.]*50, macd=[2.]*50,
                      macd_hist=[.5]*50, volume=[100.]*50, volume_sma_20=[100.]*50)).to_csv(path,index=False)
    monkeypatch.setattr(api, 'get_asset_csv_path', lambda *a: str(path))
    monkeypatch.setattr(api, 'compute_indicators', lambda df: df)
    monkeypatch.setattr(api, 'predict_for_latest_state', lambda *a: dict(
        action=0,probs=[.8,.1,.1],confidence=80.,sentiment='Neutral',price=50000.))
    monkeypatch.setattr(api, 'SHAP_CACHE', {})
    return TestClient(api.app)


@pytest.mark.parametrize('action,expected', [(0,'cash'),(1,'long'),(2,'short')])
def test_server_prediction_overrides_conflicting_client_signal(advisor,monkeypatch,action,expected):
    probs=np.full(3,.1); probs[action]=.8
    monkeypatch.setattr(api,'predict_for_latest_state',lambda *a:dict(
        action=action,probs=probs.tolist(),confidence=80.,sentiment='Neutral',price=50000.))
    body=advisor.post('/api/chat',json=dict(message='why?',asset_name='BTC',signal='SELL')).json()
    assert expected in body['reply'].lower()
    assert '50,000' in body['reply']


def test_cash_signal_does_not_invent_sideways_market(advisor):
    text=advisor.post('/api/chat',json=dict(message='why?',asset_name='BTC')).json()['reply'].lower()
    assert 'sideways' not in text and 'momentum relatively flat' not in text
    assert 'attributions' in text and 'unavailable' in text


def test_profit_question_discloses_missing_backtest(advisor):
    text=advisor.post('/api/chat',json=dict(message='What is my profit?',asset_name='BTC')).json()['reply'].lower()
    assert 'backtest' in text and ('no ' in text or 'not ' in text or "don't" in text or 'unavailable' in text)


def test_general_market_does_not_invent_cross_asset_observations(advisor):
    text=advisor.post('/api/chat',json=dict(message='How are markets?')).json()['reply'].lower()
    assert 'steady' not in text and 'sideways' not in text
    assert 'select' in text or 'unavailable' in text


def test_missing_prediction_is_disclosed(advisor,monkeypatch):
    monkeypatch.setattr(api,'get_asset_csv_path',lambda *a:None)
    text=advisor.post('/api/chat',json=dict(message='why?',asset_name='BTC',signal='BUY',price=50000.)).json()['reply'].lower()
    assert 'unavailable' in text or 'unverified' in text


@pytest.mark.parametrize('text',['Buy now.','Go short.','Open a long position.'])
def test_cash_rejects_directional_recommendations(text):
    assert not api.validate_chat_response(text,'NEUTRAL',50000.,'BTC')[0]


def test_small_price_hallucination_rejected():
    assert not api.validate_chat_response('Current price is $55,000.','BUY (LONG)',50000.,'BTC')[0]


def test_low_price_asset_hallucination_rejected():
    assert not api.validate_chat_response('Current price is $0.90.','BUY (LONG)',.10,'DOGE')[0]


def test_invalid_generative_response_uses_grounded_fallback(advisor,monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY','offline-test-only')
    class Response:
        status_code=200
        def json(self):
            return {'candidates':[{'content':{'parts':[{'text':'Buy now at $50,000.'}]}}]}
    monkeypatch.setattr(api.requests,'post',lambda *a,**k:Response())
    body=advisor.post('/api/chat',json=dict(message='why?',asset_name='BTC')).json()
    assert 'buy now' not in body['reply'].lower()
    assert 'cash' in body['reply'].lower()


def test_confidence_is_not_promised_accuracy(advisor):
    text=advisor.post('/api/chat',json=dict(message='How confident are you?',asset_name='BTC')).json()['reply'].lower()
    assert '80.0%' in text and 'not a calibrated probability of profit' in text


def test_histogram_uses_histogram_column(advisor):
    text=advisor.post('/api/chat',json=dict(message='why?',asset_name='BTC')).json()['reply']
    assert '0.5000' in text and '2.0000' not in text


def test_generation_outage_keeps_question_specific_answer(advisor,monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY','offline-test-only')
    def unavailable(*a,**k):
        raise TimeoutError('simulated outage')
    monkeypatch.setattr(api.requests,'post',unavailable)
    body=advisor.post('/api/chat',json=dict(message='Explain RSI',asset_name='BTC')).json()
    assert 'rsi' in body['reply'].lower() and '80.0' in body['reply']
    assert 'generation' in body and body['generation']=='deterministic'


def test_valid_generation_preserved(advisor,monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY','offline-test-only')
    class Response:
        status_code=200
        def json(self):
            return {'candidates':[{'content':{'parts':[{'text':'The target is Cash. The available price is $50,000.00.'}]}}]}
    monkeypatch.setattr(api.requests,'post',lambda *a,**k:Response())
    body=advisor.post('/api/chat',json=dict(message='What is the signal?',asset_name='BTC')).json()
    assert body['validation']=='passed'


def test_unmatched_shap_cache_is_not_used(advisor,monkeypatch):
    monkeypatch.setattr(api,'SHAP_CACHE',{'BTC_1H':({'action':1,'shap_values':np.ones((1,17,3)).tolist()},0)})
    text=advisor.post('/api/chat',json=dict(message='why?',asset_name='BTC')).json()['reply'].lower()
    assert 'attributions' in text and 'unavailable' in text


def test_matching_shap_cache_is_used_for_current_decision(advisor,monkeypatch):
    monkeypatch.setattr(api,'predict_for_latest_state',lambda *a:dict(
        action=1,probs=[.1,.8,.1],confidence=80.,sentiment='Bullish',price=50000.,state_id='state-1'))
    values=np.zeros((1,8*api.obs_dim,3))
    values[0,0,1]=.2
    monkeypatch.setattr(api,'SHAP_CACHE',{
        'BTC_1H':({'action':1,'state_id':'state-1','shap_values':values.tolist()},time.time())})
    text=advisor.post('/api/chat',json=dict(message='why?',asset_name='BTC')).json()['reply'].lower()
    assert 'open ratio' in text
    assert 'attributions are currently unavailable' not in text
