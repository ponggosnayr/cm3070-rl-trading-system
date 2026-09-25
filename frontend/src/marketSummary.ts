interface SummaryAsset {
  label: string;
  price?: number;
  change_pct?: number;
  signal?: string;
  action?: number;
  regime?: string;
  quote_is_live?: boolean;
  candle_is_live?: boolean;
  model_eval_time?: string;
  quote_time?: string;
}

export function targetLabel(signal?: string, action?: number): string {
  if (action !== undefined && [0, 1, 2].includes(action)) return ['Cash', 'Long', 'Short'][action];
  const value = signal?.toUpperCase();
  if (value === 'BUY' || value === 'LONG') return 'Long';
  if (value === 'SELL' || value === 'SHORT') return 'Short';
  if (value === 'NEUTRAL' || value === 'CASH' || value === 'HOLD') return 'Cash';
  return 'Unavailable';
}

export function summarizeMarket(assets: SummaryAsset[]): string {
  if (!assets.length) return 'Market data is not available yet.';
  const known = assets.filter(a => typeof a.change_pct === 'number' && Number.isFinite(a.change_pct));
  const up = known.filter(a => a.change_pct! > 0);
  const down = known.filter(a => a.change_pct! < 0);
  let market = !known.length ? 'Price movements are unavailable in the loaded data.'
    : down.length === assets.length ? 'Loaded price changes show all tracked assets lower.'
    : up.length === assets.length ? 'Loaded price changes show all tracked assets higher.'
    : 'In the loaded data, ' + up.length + (up.length === 1 ? ' asset is higher, ' : ' assets are higher, ') + down.length + ' lower and ' + (known.length - up.length - down.length) + ' unchanged.';
  const crypto = assets.filter(a => /\((BTC|ETH|DOGE)\)$/.test(a.label));
  const etfs = assets.filter(a => /\((SPY|QQQ)\)$/.test(a.label));
  if (crypto.length && etfs.length && crypto.length + etfs.length === assets.length
    && crypto.every(a => known.includes(a) && a.change_pct! < 0)
    && etfs.every(a => known.includes(a) && a.change_pct! > 0)) {
    market = 'Loaded price changes show crypto lower, while stock ETFs are higher.';
  }
  if (known.length && known.length !== assets.length) market += ' Some price changes are unavailable.';
  const liveQuotes = assets.filter(a => a.quote_is_live === true).length;
  const liveCandles = assets.filter(a => a.candle_is_live === true).length;
  const quoteStatus = liveQuotes === assets.length ? 'Quotes are live.'
    : liveQuotes === 0 ? 'Quotes are cached or archived.'
    : `${liveQuotes} of ${assets.length} quotes are live; the rest are cached or archived.`;
  const candleStatus = liveCandles === assets.length ? '5m candles are live.'
    : liveCandles === 0 ? '5m candles are offline or archived.'
    : `${liveCandles} of ${assets.length} 5m candle feeds are live; the rest are offline or archived.`;
  market += ` ${quoteStatus} ${candleStatus}`;
  if (down.length) {
    const weakest = down.reduce((a, b) => a.change_pct! <= b.change_pct! ? a : b);
    market += ' In this loaded snapshot, ' + weakest.label + ' has the largest decline, down ' + Math.abs(weakest.change_pct!).toFixed(2) + '%.';
  }
  const targets = assets.map(a => targetLabel(a.signal, a.action));
  const count = (target: string) => targets.filter(t => t === target).length;
  const parts = [
    count('Long') ? 'holding ' + count('Long') + (count('Long') === 1 ? ' asset' : ' assets') : '',
    count('Short') ? 'short positions in ' + count('Short') + (count('Short') === 1 ? ' asset' : ' assets') : '',
    count('Cash') ? 'staying in cash for ' + count('Cash') + (count('Cash') === 1 ? ' asset' : ' assets') : '',
  ].filter(Boolean);
  const view = count('Long') === assets.length
    ? 'The loaded model snapshot favours holding all ' + assets.length + ' tracked assets.'
    : parts.length ? 'The loaded model snapshot favours ' + parts.join(', ') + '.' : 'Model recommendations are unavailable.';
  const evalTimes = [...new Set(assets.map(a => a.model_eval_time).filter((t): t is string => Boolean(t)))];
  const snapshotTime = evalTimes.length === 1 ? ` Model evaluation time: ${evalTimes[0]}.`
    : evalTimes.length > 1 ? ' Model evaluation times vary by asset.' : '';
  const missing = parts.length && count('Unavailable') ? ' Recommendations for ' + count('Unavailable') + ' assets are unavailable.' : '';
  return market + '\n\nAI outlook: ' + view + missing + snapshotTime + '\n\nSelect an asset to review its loaded recommendation, price history and risks.';
}

export function summarizeAsset(asset: SummaryAsset): string {
  const price = typeof asset.price === 'number' && Number.isFinite(asset.price) && asset.price > 0
    ? asset.price.toLocaleString('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: asset.price < 1 ? 5 : 2 }) : 'unavailable';
  const change = typeof asset.change_pct === 'number' && Number.isFinite(asset.change_pct)
    ? `${asset.change_pct >= 0 ? '+' : ''}${asset.change_pct.toFixed(2)}%` : 'unavailable';
  const target = targetLabel(asset.signal, asset.action);
  const regime = asset.regime?.trim() || 'Unavailable';
  const quoteStatus = asset.quote_is_live === true ? 'live quote' : 'cached or archived quote';
  const candleStatus = asset.candle_is_live === true ? 'live 5m candles' : 'offline or archived 5m candles';
  const quoteTime = asset.quote_time ? ` Quote time: ${asset.quote_time}.` : '';
  const evalTime = asset.model_eval_time ? ` Model evaluation time: ${asset.model_eval_time}.` : '';
  return `${asset.label} loaded price: ${price} (${change}) from a ${quoteStatus}; chart data: ${candleStatus}. The loaded advisor snapshot targets ${target} within a ${regime} market regime.${quoteTime}${evalTime}`;
}
