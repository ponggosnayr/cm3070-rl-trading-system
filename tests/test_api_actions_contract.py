import pytest
from fastapi.testclient import TestClient
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api import app, predict_for_latest_state

# Enforce deterministic offline execution for automated tests per Rule 18
os.environ["GEMINI_API_KEY"] = ""

from src.xai_engine import XAIEngine

client = TestClient(app)

class TestActionContractAPI:
    def test_root_health(self):
        res = client.get('/')
        assert res.status_code == 200
        assert res.json()['status'] == 'ok'
    
    def test_action_2_short_does_not_return_profit_taking(self):
        """
        Prior bug reproduction: Action 2 / SELL was mapped to EXIT LONG ('profit-taking').
        In the canonical Discrete(3) contract, action 2 must represent SHORT exposure.
        """
        payload = {
            'message': 'why did you make this trade?',
            'asset_name': 'BTC',
            'price': 50000.0,
            'signal': 'SELL',
            'confidence': 88.0,
            'sentiment': 'Bearish'
        }
        res = client.post('/api/chat', json=payload)
        assert res.status_code == 200
        reply = res.json().get('reply', '')
        # Verify it does NOT claim to take profits or exit a long
        assert 'profit-taking' not in reply.lower()
        assert 'closing your long position' not in reply.lower()
        # Verify it explicitly recognizes SHORT
        assert 'short' in reply.lower()
    
    def test_action_1_buy_returns_long_advice(self):
        payload = {
            'message': 'why did you make this trade?',
            'asset_name': 'BTC',
            'price': 50000.0,
            'signal': 'BUY',
            'confidence': 90.0,
            'sentiment': 'Bullish'
        }
        res = client.post('/api/chat', json=payload)
        assert res.status_code == 200
        reply = res.json().get('reply', '')
        assert 'long' in reply.lower() or 'buy' in reply.lower()
    
    def test_action_0_neutral_returns_cash_advice(self):
        payload = {
            'message': 'why did you make this trade?',
            'asset_name': 'BTC',
            'price': 50000.0,
            'signal': 'NEUTRAL',
            'confidence': 75.0,
            'sentiment': 'Neutral'
        }
        res = client.post('/api/chat', json=payload)
        assert res.status_code == 200
        reply = res.json().get('reply', '')
        assert 'neutral' in reply.lower() or 'cash' in reply.lower()
    
    def test_invalid_signal_graceful_fallback(self):
        payload = {
            'message': 'explain yourself',
            'asset_name': 'BTC',
            'price': 50000.0,
            'signal': 'INVALID_SIGNAL_XYZ',
            'confidence': 50.0,
            'sentiment': 'Neutral'
        }
        res = client.post('/api/chat', json=payload)
        assert res.status_code == 200
        reply = res.json().get('reply', '')
        assert len(reply) > 0

    def test_predict_latest_state_contract(self):
        pred = predict_for_latest_state('Bitcoin (BTC)', '1H')
        assert pred is not None
        assert pred['action'] in (0, 1, 2)
        assert len(pred['probs']) == 3
        assert pred['signal'] in ('NEUTRAL', 'BUY', 'SELL')
