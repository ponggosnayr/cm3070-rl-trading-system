import { test } from 'node:test';
import assert from 'node:assert/strict';
import { summarizeMarket, summarizeAsset, targetLabel } from './marketSummary.ts';

test('falling prices with all Long targets separates market direction from the model view', () => {
  const text = summarizeMarket(Array.from({length:5}, (_,i) => ({label:String(i),change_pct:-1,signal:'BUY',regime:'Sideways / Quiet'})));
  assert.match(text, /tracked assets are trading lower/);
  assert.match(text, /favours holding all 5 tracked assets/);
  assert.doesNotMatch(text, /rebound|recovery|protect capital/);
});

test('mixed market conditions return a balanced briefing', () => {
  const text = summarizeMarket([
    {label:'a',change_pct:2,action:2},
    {label:'b',change_pct:0,action:0},
    {label:'c',change_pct:-1,action:1}
  ]);
  assert.match(text, /1 asset is higher, 1 lower and 1 unchanged/);
  assert.match(text, /holding 1 asset, short positions in 1 asset, staying in cash for 1 asset/);
});

test('missing data returns unavailable gracefully', () => {
  assert.equal(summarizeMarket([]), 'Market data is not available yet.');
  assert.equal(targetLabel(), 'Unavailable');
  assert.match(summarizeAsset({label:'a'}), /Unavailable/);
});

test('asset summary provides clean natural advisory overview', () => {
  const text = summarizeAsset({label:'BTC',price:100,change_pct:-3,signal:'SELL',action:1,regime:'Sideways / Quiet'});
  assert.match(text, /BTC is trading at \$100\.00 \(-3\.00%\)\./);
  assert.match(text, /The advisor target is Long within a Sideways \/ Quiet market regime\./);
});

 test('briefing reflects screenshot data and handles missing inputs', () => {
  const text = summarizeMarket([
    {label:'Bitcoin (BTC)',change_pct:-0.93,action:1},
    {label:'Ethereum (ETH)',change_pct:-1.43,action:1},
    {label:'Dogecoin (DOGE)',change_pct:-3.17,action:1},
    {label:'S&P 500 (SPY)',change_pct:0.13,action:1},
    {label:'NASDAQ 100 (QQQ)',change_pct:0.35,action:1},
  ]);
  assert.match(text, /Crypto is trading lower, while stock ETFs are posting gains/);
  assert.match(text, /Dogecoin \(DOGE\).*down 3.17%/);
  assert.match(summarizeMarket([{label:'Unknown'}]), /Price movements are currently unavailable/);
  assert.match(summarizeMarket([{label:'Unknown'}]), /Model recommendations are currently unavailable/);
});
