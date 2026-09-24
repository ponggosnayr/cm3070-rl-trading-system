interface SummaryAsset {
  label: string;
  price?: number;
  change_pct?: number;
  signal?: string;
  action?: number;
  regime?: string;
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
  let market = !known.length ? 'Price movements are currently unavailable.'
    : down.length === assets.length ? 'Your tracked assets are trading lower.'
    : up.length === assets.length ? 'Your tracked assets are trading higher.'
    : 'Across available prices, ' + up.length + (up.length === 1 ? ' asset is higher, ' : ' assets are higher, ') + down.length + ' lower and ' + (known.length - up.length - down.length) + ' unchanged.';
  const crypto = assets.filter(a => /\((BTC|ETH|DOGE)\)$/.test(a.label));
  const etfs = assets.filter(a => /\((SPY|QQQ)\)$/.test(a.label));
  if (crypto.length && etfs.length && crypto.length + etfs.length === assets.length
    && crypto.every(a => known.includes(a) && a.change_pct! < 0)
    && etfs.every(a => known.includes(a) && a.change_pct! > 0)) {
    market = 'Crypto is trading lower, while stock ETFs are posting gains.';
  }
  if (known.length && known.length !== assets.length) market += ' Some price changes are unavailable.';
  if (down.length) {
    const weakest = down.reduce((a, b) => a.change_pct! <= b.change_pct! ? a : b);
    market += ' ' + weakest.label + ' has the largest decline in your watchlist, down ' + Math.abs(weakest.change_pct!).toFixed(2) + '%.';
  }
  const targets = assets.map(a => targetLabel(a.signal, a.action));
  const count = (target: string) => targets.filter(t => t === target).length;
  const parts = [
    count('Long') ? 'holding ' + count('Long') + (count('Long') === 1 ? ' asset' : ' assets') : '',
    count('Short') ? 'short positions in ' + count('Short') + (count('Short') === 1 ? ' asset' : ' assets') : '',
    count('Cash') ? 'staying in cash for ' + count('Cash') + (count('Cash') === 1 ? ' asset' : ' assets') : '',
  ].filter(Boolean);
  const view = count('Long') === assets.length
    ? 'The model currently favours holding all ' + assets.length + ' tracked assets.'
    : parts.length ? 'The model currently favours ' + parts.join(', ') + '.' : 'Model recommendations are currently unavailable.';
  const missing = parts.length && count('Unavailable') ? ' Recommendations for ' + count('Unavailable') + ' assets are unavailable.' : '';
  return market + '\n\nAI outlook: ' + view + missing + '\n\nExplore your next move: Select an asset to review its recommendation, recent performance and risks.';
}

export function summarizeAsset(asset: SummaryAsset): string {
  const price = typeof asset.price === 'number' && Number.isFinite(asset.price) && asset.price > 0
    ? asset.price.toLocaleString('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: asset.price < 1 ? 5 : 2 }) : 'unavailable';
  const change = typeof asset.change_pct === 'number' && Number.isFinite(asset.change_pct)
    ? `${asset.change_pct >= 0 ? '+' : ''}${asset.change_pct.toFixed(2)}%` : 'unavailable';
  const target = targetLabel(asset.signal, asset.action);
  const regime = asset.regime?.trim() || 'Unavailable';
  return `${asset.label} is trading at ${price} (${change}). The advisor target is ${target} within a ${regime} market regime.`;
}
