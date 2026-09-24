import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequestCache } from './requestCache.ts';

test('navigation and refresh share one unfinished calculation', async () => {
  const cache = createRequestCache(1000);
  let complete, calls = 0;
  const load = () => { calls++; return new Promise(resolve => { complete = resolve; }); };
  const first = cache.get('btc', load);
  const second = cache.get('btc', load, true);
  assert.equal(first, second);
  await Promise.resolve();
  complete({ roi: -12.88 });
  await first;
  assert.deepEqual(await cache.get('btc', load), { roi: -12.88 });
  assert.equal(calls, 1);
});

test('a slow backtest does not block SHAP or another asset', async () => {
  const cache = createRequestCache(1000);
  let complete;
  const slow = cache.get('backtest/btc', () => new Promise(resolve => { complete = resolve; }));
  assert.equal(await cache.get('shap/btc', async () => 'drivers'), 'drivers');
  assert.equal(await cache.get('backtest/eth', async () => 'eth'), 'eth');
  complete('btc');
  assert.equal(await slow, 'btc');
});

test('expired entries and explicit refresh fetch again', async () => {
  let time = 0, calls = 0;
  const cache = createRequestCache(100, () => time);
  const load = async () => ++calls;
  assert.equal(await cache.get('btc', load), 1);
  assert.equal(await cache.get('btc', load, true), 2);
  time = 101;
  assert.equal(cache.peek('btc'), undefined);
  assert.equal(await cache.get('btc', load), 3);
});

test('failed requests can be retried and are not cached', async () => {
  const cache = createRequestCache(1000);
  await assert.rejects(cache.get('btc', async () => { throw new Error('offline'); }));
  assert.equal(cache.peek('btc'), undefined);
  assert.equal(await cache.get('btc', async () => 'recovered'), 'recovered');
});
