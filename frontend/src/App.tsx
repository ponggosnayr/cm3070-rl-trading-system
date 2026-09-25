import React, { useState, useEffect, useRef } from 'react';
import DOMPurify from 'dompurify';
import { createRequestCache, fetchJson } from './requestCache';
import RecommendationCard from './RecommendationCard';
import { summarizeMarket, summarizeAsset, targetLabel } from './marketSummary';
import { Bot, Activity, Send, ArrowRight, TrendingUp, TrendingDown, Minus, BookOpen, BarChart2, ArrowLeft, Cpu, ChevronDown, ChevronUp, Radio, ChevronRight, ChevronLeft, Sparkles, RotateCcw, LayoutDashboard, Search, Shield } from 'lucide-react';
import { ResponsiveContainer, ComposedChart, Line, XAxis, YAxis, Tooltip as RechartsTooltip, AreaChart, Area, Scatter, CartesianGrid } from 'recharts';

interface Candle {
  open: number;
  high: number;
  low: number;
  close: number;
  time?: string;
  timestamp?: number;
}

interface AssetData {
  label: string;
  price: number;
  signal: string;
  confidence: number;
  sentiment: string;
  history: number[];
  candles?: Candle[];
  regime?: string;
  change?: number;
  change_pct?: number;
  summary?: string;
  probs?: number[];
  action?: number;
  model_eval_time?: string;
  quote_time?: string;
  candle_source?: string;
  is_synthetic_candles?: boolean;
  quote_is_live?: boolean;
  candle_is_live?: boolean;
  last_candle_time?: string | null;
}

interface TradeItem {
  reason?: string;
  policy_target?: number;
  step: number;
  action: number;
  price: number;
  net_worth: number;
}

interface BacktestMetrics {
  roi: number;
  bh_return: number;
  sharpe: number;
  sortino: number;
  max_dd: number;
  calmar: number;
  trades: number;
  final_capital: number;
  history: number[];
  trade_history: TradeItem[];
  dates?: string[];
  regime?: string;
  regime_title?: string;
  sample_type?: string;
}

interface ShapItem {
  feature: string;
  value: number;
  importance_pct?: number;
}

interface ShapData {
  shap: ShapItem[];
  action: number;
}

interface WalkForwardRow {
  Fold: number;
  'Train Range'?: string;
  'Test Range': string;
  'Out-of-Sample ROI (%)': number;
  'Market B&H (%)': number;
  'Out-of-Sample Sharpe': number;
  'Max DD (%)': number;
  Trades: number;
}

const WFV_ERA_MAP: Record<number, { era: string; regime: string; regimeColor: string; description: string }> = {
  1: { era: '2020–2021 Early Cycle', regime: 'Accumulation Rally', regimeColor: '#38bdf8', description: 'Early post-crash expansion and baseline recovery' },
  2: { era: '2021–2022 Crypto Winter', regime: 'Severe Bear Crash', regimeColor: '#f87171', description: 'Deleveraging shock & systemic liquidity unwind' },
  3: { era: '2022–2023 Bull Recovery', regime: 'Parabolic Rebound', regimeColor: '#4ade80', description: 'High-momentum short squeeze & cycle pivot' },
  4: { era: '2023–2024 Market Expansion', regime: 'Institutional Trend Bull', regimeColor: '#a78bfa', description: 'Spot ETF momentum & sustained institutional flow' },
  5: { era: '2025–2026 Late Cycle Volatility', regime: 'Choppy Bear Correction', regimeColor: '#fbbf24', description: 'Macro tightening & cyclic distribution' },
};

const SHAP_FEATURE_CONFIG: Record<string, { label: string; icon: string }> = {
  "Momentum (ROC24)": { label: "24h Price Momentum", icon: "🚀" },
  "Short Trend (SMA20)": { label: "20-Period Trend (SMA20)", icon: "📈" },
  "Medium Trend (SMA99)": { label: "Long-Term Trend (SMA99)", icon: "📊" },
  "Normalized Volume": { label: "Trading Volume Flow", icon: "🌊" },
  "Volatility": { label: "Intraday Volatility Check", icon: "🛡️" },
  "Volatility Ratio": { label: "Volatility Spread", icon: "⚡" },
  "Price Change": { label: "Latest Candle Momentum", icon: "⚡" },
  "Price Change Prev": { label: "Prior Candle Velocity", icon: "⏱️" },
  "Open Ratio": { label: "Open Price Alignment", icon: "🎯" },
  "High Ratio": { label: "Selling Pressure at Highs", icon: "🔼" },
  "Low Ratio": { label: "Buyers Stepping In at Lows", icon: "🔽" },
  "Lower Wick Support": { label: "Buyers Stepping In at Lows", icon: "🔽" },
  "Lower Wick": { label: "Buyers Stepping In at Lows", icon: "🔽" },
  "Upper Wick Pressure": { label: "Selling Pressure at Highs", icon: "🔼" },
  "Upper Wick": { label: "Selling Pressure at Highs", icon: "🔼" },
  "Macro Drawdown": { label: "Macro Drawdown Distance", icon: "📉" },
  "Market Regime": { label: "Broader Market Trends", icon: "🌐" },
  "Macro Regime State": { label: "Broader Market Trends", icon: "🌐" },
  "Idle Steps": { label: "Position Inactivity Cooldown", icon: "⏳" },
  "Portfolio Position": { label: "Current Position Exposure", icon: "💼" },
  "Unrealized P&L": { label: "Open Trade Profit/Loss", icon: "💰" },
  "Holding Duration": { label: "Trade Time in Market", icon: "⏱️" },
  "RSI (Momentum)": { label: "Relative Strength Momentum", icon: "🚀" },
  "MACD (Trend)": { label: "Moving Average Momentum", icon: "📈" },
};

function cleanFeatureLabel(featureName: string): { label: string; icon: string } {
  if (SHAP_FEATURE_CONFIG[featureName]) return SHAP_FEATURE_CONFIG[featureName];
  const cleaned = featureName
    .replace(/lower wick( support)?/gi, "Buyers Stepping In at Lows")
    .replace(/upper wick( pressure)?/gi, "Selling Pressure at Highs")
    .replace(/macro regime state/gi, "Broader Market Trends")
    .replace(/signal flipping/gi, "avoiding frequent trades to save on fees")
    .replace(/turnover/gi, "trading fees");
  return { label: cleaned, icon: "🔍" };
}

interface MonteCarloSummary {
  mean_roi: number;
  std_roi: number;
  min_roi: number;
  max_roi: number;
  var_5th: number;
  mean_sharpe: number;
  mean_drawdown: number;
  success_rate: number;
}

interface MonteCarloData {
  summary: MonteCarloSummary;
  percentiles: {
    '5': number[];
    '25': number[];
    '50': number[];
    '75': number[];
    '95': number[];
  };
  sample_paths: number[][];
  step_count: number;
  dates?: string[];
}

interface ComparisonRow {
  Model: string;
  'ROI (%)'?: number;
  ROI?: number;
  'Max DD (%)'?: number;
  Max_DD?: number;
  Sharpe?: number;
  Trades?: number;
}


interface ChatPayload {
  message: string;
  history: Array<{ role: string; content: string }>;
  asset_name?: string;
  price?: number;
  signal?: string;
  confidence?: number;
  sentiment?: string;
}

// Simple markdown-to-HTML converter supporting headings, lists, bold, and italics with DOMPurify sanitization
function renderMarkdown(text: string) {
  let html = text;

  // Parse headings (must run before converting newlines to preserve line boundaries)
  html = html.replace(/^### (.*?)$/gm, '<h4 style="margin: 0.8rem 0 0.4rem 0; color: #FFFFFF; font-size: 1.05rem; font-weight: bold;">$1</h4>');
  html = html.replace(/^## (.*?)$/gm, '<h3 style="margin: 1rem 0 0.5rem 0; color: #FFFFFF; font-size: 1.15rem; font-weight: bold;">$1</h3>');
  html = html.replace(/^# (.*?)$/gm, '<h2 style="margin: 1.2rem 0 0.6rem 0; color: #FFFFFF; font-size: 1.25rem; font-weight: bold;">$1</h2>');

  // Bold and Italics
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

  // Bullet lists (lines starting with * or - followed by a space)
  html = html.replace(/^\s*[*\\-]\s+(.*?)$/gm, '<li style="margin: 0.25rem 0 0.25rem 1.25rem; list-style-type: disc; line-height: 1.5;">$1</li>');

  // Group consecutive list items into ul tags
  html = html.replace(/(<li style=".*?">.*?<\/li>)+/g, '<ul style="margin: 0.5rem 0; padding-left: 0;">$&</ul>');

  // Convert remaining newlines to br
  html = html.replace(/\n/g, '<br/>');

  // Clean up any double br around block elements
  html = html.replace(/<br\/>(<\/?(?:ul|li|h2|h3|h4))/g, '$1');
  html = html.replace(/(<\/(?:ul|li|h2|h3|h4)>)<br\/>/g, '$1');

  return DOMPurify.sanitize(html);
}

function getAssetDayTrend(asset?: AssetData | null): { isUp: boolean; changePct: number; color: string } {
  if (!asset) return { isUp: true, changePct: 0, color: '#34D399' };
  
  if (typeof asset.change_pct === 'number' && !isNaN(asset.change_pct)) {
    const isUp = asset.change_pct >= 0;
    return {
      isUp,
      changePct: asset.change_pct,
      color: isUp ? '#34D399' : '#F87171'
    };
  }

  let firstPrice = 0;
  let lastPrice = asset.price || 0;

  if (asset.candles && asset.candles.length > 0) {
    firstPrice = typeof asset.candles[0].open === 'number' ? asset.candles[0].open : asset.candles[0].close;
    lastPrice = typeof asset.candles[asset.candles.length - 1].close === 'number' ? asset.candles[asset.candles.length - 1].close : lastPrice;
  } else if (asset.history && asset.history.length > 1) {
    firstPrice = asset.history[0];
    lastPrice = asset.history[asset.history.length - 1];
  }

  if (!firstPrice || firstPrice <= 0) {
    return { isUp: true, changePct: 0, color: '#34D399' };
  }

  const diff = lastPrice - firstPrice;
  const pct = (diff / firstPrice) * 100;
  const isUp = diff >= 0;
  return {
    isUp,
    changePct: pct,
    color: isUp ? '#34D399' : '#F87171'
  };
}

// SVG sparkline component
export function Sparkline({ data, color, width = 120, height = 40 }: { data: number[]; color?: string; width?: number; height?: number }) {
  if (!data || data.length < 2) return null;
  const isUp = data[data.length - 1] >= data[0];
  const strokeColor = color || (isUp ? '#34D399' : '#F87171');
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const uid = `grad-${strokeColor.replace(/[^a-zA-Z0-9]/g, '')}-${width}`;
  const points = data.map((v, i) => {
    const x = (i / (data.length - 1)) * width;
    const y = height - ((v - min) / range) * (height - 4) - 2;
    return `${x},${y}`;
  }).join(' ');

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ display: 'block', margin: '0.5rem auto 0' }}>
      <defs>
        <linearGradient id={uid} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={strokeColor} stopOpacity="0.3" />
          <stop offset="100%" stopColor={strokeColor} stopOpacity="0" />
        </linearGradient>
      </defs>
      <polygon points={`0,${height} ${points} ${width},${height}`} fill={`url(#${uid})`} />
      <polyline points={points} fill="none" stroke={strokeColor} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function getSmoothPath(pts: { x: number; y: number }[]): string {
  if (pts.length === 0) return '';
  if (pts.length === 1) return `M ${pts[0].x.toFixed(1)},${pts[0].y.toFixed(1)}`;
  if (pts.length === 2) return `M ${pts[0].x.toFixed(1)},${pts[0].y.toFixed(1)} L ${pts[1].x.toFixed(1)},${pts[1].y.toFixed(1)}`;

  let path = `M ${pts[0].x.toFixed(1)},${pts[0].y.toFixed(1)}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[Math.max(0, i - 1)];
    const p1 = pts[i];
    const p2 = pts[i + 1];
    const p3 = pts[Math.min(pts.length - 1, i + 2)];

    const tension = 0.2;
    const cp1x = p1.x + ((p2.x - p0.x) * tension);
    const cp1y = p1.y + ((p2.y - p0.y) * tension);
    const cp2x = p2.x - ((p3.x - p1.x) * tension);
    const cp2y = p2.y - ((p3.y - p1.y) * tension);

    path += ` C ${cp1x.toFixed(1)},${cp1y.toFixed(1)} ${cp2x.toFixed(1)},${cp2y.toFixed(1)} ${p2.x.toFixed(1)},${p2.y.toFixed(1)}`;
  }
  return path;
}

function formatCandleDateTime(c?: Candle): string {
  if (!c) return '';
  if (typeof c.timestamp === 'number' && c.timestamp > 0) {
    // Auto-detect seconds vs milliseconds (Unix seconds ~1.7e9, ms ~1.7e12)
    const tsMs = c.timestamp < 1e11 ? c.timestamp * 1000 : c.timestamp;
    const flooredTs = Math.floor(tsMs / 300000) * 300000;
    const d = new Date(flooredTs);
    if (!isNaN(d.getTime())) {
      return d.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric'
      }) + ', ' + d.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        hour12: false
      });
    }
  }
  if (c.time) {
    const d = new Date(c.time);
    if (!isNaN(d.getTime())) {
      const flooredTs = Math.floor(d.getTime() / 300000) * 300000;
      const flooredDate = new Date(flooredTs);
      return flooredDate.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric'
      }) + ', ' + flooredDate.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        hour12: false
      });
    }
    return String(c.time);
  }
  return '';
}

export function AreaLineChart({ 
  data, 
  candles, 
  color, 
  height = 70, 
  interactive = false,
  isLive,
  candleSource
}: { 
  data?: number[]; 
  candles?: Candle[]; 
  color?: string; 
  height?: number; 
  interactive?: boolean; 
  isLive?: boolean;
  candleSource?: string;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [containerWidth, setContainerWidth] = useState<number>(0);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const el = containerRef.current;
    const update = () => {
      if (el && el.clientWidth > 0) setContainerWidth(el.clientWidth);
    };
    update();
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        if (entry.contentRect.width > 0) {
          setContainerWidth(Math.round(entry.contentRect.width));
        }
      }
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const series: number[] = React.useMemo(() => {
    if (candles && candles.length > 0) return candles.map(c => c.close);
    if (data && data.length > 1) return data;
    return [];
  }, [data, candles]);

  const priceTicks = React.useMemo(() => {
    if (!interactive || !series || series.length < 2) return [];
    const min = Math.min(...series);
    const max = Math.max(...series);
    const r = max - min || 1;
    const padY = 14;
    const pH = height - padY * 2;
    const ticks = [];
    const count = 4;
    for (let i = 0; i <= count; i++) {
      const val = min + (r * i) / count;
      const y = height - padY - ((val - min) / r) * pH;
      ticks.push({ val, y });
    }
    return ticks;
  }, [series, interactive, height]);

  if (!series || series.length < 2) {
    return <div ref={containerRef} style={{ color: 'var(--text-secondary)', fontSize: '0.75rem', textAlign: 'center', padding: '1rem' }}>Loading price chart...</div>;
  }

  const startPrice = (candles && candles.length > 0 && typeof candles[0].open === 'number') 
    ? candles[0].open 
    : series[0];
  const endPrice = series[series.length - 1];
  const isUp = endPrice >= startPrice;
  const strokeColor = color || (isUp ? '#34D399' : '#F87171');
  const minVal = Math.min(...series);
  const maxVal = Math.max(...series);
  const range = maxVal - minVal || 1;

  // Responsive width from container element (fallback to 300 / 900 until first measurement)
  const rightGutter = interactive ? 65 : 0;
  const width = containerWidth > 0 ? containerWidth : (interactive ? 900 : 300);
  const plotWidth = Math.max(100, width - rightGutter);
  const paddingY = interactive ? 14 : 8;
  const plotH = height - paddingY * 2;

  const points = series.map((val, i) => {
    const x = (i / (series.length - 1)) * plotWidth;
    const y = height - paddingY - ((val - minVal) / range) * plotH;
    return { x, y, val };
  });

  const smoothLinePath = getSmoothPath(points);
  const firstPt = points[0];
  const lastPt = points[points.length - 1];
  const smoothAreaPath = `${smoothLinePath} L ${lastPt.x.toFixed(1)},${height} L ${firstPt.x.toFixed(1)},${height} Z`;
  const gradId = `area-grad-${strokeColor.replace(/[^a-zA-Z0-9]/g, '')}-${height}`;

  const activePoint = hoverIdx !== null && hoverIdx >= 0 && hoverIdx < points.length ? points[hoverIdx] : points[points.length - 1];
  const activeCandle = candles && candles.length > 0 
    ? (hoverIdx !== null && hoverIdx >= 0 && hoverIdx < candles.length ? candles[hoverIdx] : candles[candles.length - 1]) 
    : undefined;

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!interactive) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    if (mouseX < 0 || mouseX > plotWidth) return;
    const ratio = Math.max(0, Math.min(1, mouseX / plotWidth));
    const idx = Math.round(ratio * (series.length - 1));
    setHoverIdx(idx);
  };

  const pointChangePct = startPrice > 0 ? ((activePoint.val - startPrice) / startPrice) * 100 : 0;
  const isPointUp = pointChangePct >= 0;
  const yStart = height - paddingY - ((startPrice - minVal) / range) * plotH;

  return (
    <div ref={containerRef} style={{ width: '100%', position: 'relative' }}>
      {interactive && (
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem', fontSize: '0.75rem', fontFamily: "'JetBrains Mono', monospace", borderBottom: '1px solid rgba(255, 255, 255, 0.06)', paddingBottom: '0.3rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span style={{ color: 'var(--text-muted)' }}>Price:</span>
            <span style={{ color: strokeColor, fontWeight: 700 }}>
              ${activePoint.val.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: activePoint.val < 1 ? 5 : 2 })}
            </span>
            <span style={{ fontSize: '0.75rem', fontWeight: 600, color: isPointUp ? '#34D399' : '#F87171' }}>
              ({isPointUp ? '+' : ''}{pointChangePct.toFixed(2)}%)
            </span>
          </div>
          <span style={{ color: '#FFFFFF', fontWeight: 600, display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
            {hoverIdx !== null && activeCandle ? (
              <>
                <span style={{ color: 'var(--text-muted)' }}>Time:</span>
                <span style={{ color: '#FAFAFA' }}>{formatCandleDateTime(activeCandle)}</span>
              </>
            ) : isLive ? (
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', color: '#34D399' }}>
                <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: '#34D399', boxShadow: '0 0 6px #34D399', display: 'inline-block' }}></span>
                Live 5m Stream
              </span>
            ) : (
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', color: '#FBBF24' }}>
                <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: '#FBBF24', display: 'inline-block' }}></span>
                {candleSource === 'offline_historical_replay' ? 'Historical Replay (Offline)' : 'Cached 5m Stream'}
              </span>
            )}
          </span>
        </div>
      )}
      <svg
        width="100%"
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        style={{ display: 'block', overflow: 'visible', cursor: interactive ? 'crosshair' : 'default' }}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoverIdx(null)}
      >
        <defs>
          <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={strokeColor} stopOpacity="0.26" />
            <stop offset="65%" stopColor={strokeColor} stopOpacity="0.04" />
            <stop offset="100%" stopColor={strokeColor} stopOpacity="0.0" />
          </linearGradient>
        </defs>

        {/* Horizontal Gridlines & Price Ticks on Right Axis */}
        {interactive && priceTicks.map((tick, i) => (
          <g key={i}>
            <line
              x1={0}
              y1={tick.y}
              x2={plotWidth}
              y2={tick.y}
              stroke="rgba(255, 255, 255, 0.07)"
              strokeWidth={1}
              strokeDasharray="3,3"
              vectorEffect="non-scaling-stroke"
            />
            <text
              x={plotWidth + 8}
              y={tick.y + 3.5}
              fill="#71717A"
              fontSize="10"
              fontFamily="JetBrains Mono, monospace"
            >
              ${tick.val.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: tick.val < 1 ? 4 : 0 })}
            </text>
          </g>
        ))}

        {/* Right-axis separator line */}
        {interactive && (
          <line
            x1={plotWidth}
            y1={paddingY}
            x2={plotWidth}
            y2={height - paddingY}
            stroke="rgba(255, 255, 255, 0.12)"
            strokeWidth={1}
            vectorEffect="non-scaling-stroke"
          />
        )}

        {/* Day Open Dotted Baseline */}
        {interactive && startPrice >= minVal && startPrice <= maxVal && (
          <line
            x1={0}
            y1={yStart}
            x2={plotWidth}
            y2={yStart}
            stroke="rgba(255, 255, 255, 0.16)"
            strokeWidth={1}
            strokeDasharray="4,4"
            vectorEffect="non-scaling-stroke"
          />
        )}

        {/* Gradient Area Fill */}
        <path d={smoothAreaPath} fill={`url(#${gradId})`} />

        {/* Smooth Line Curve */}
        <path
          d={smoothLinePath}
          fill="none"
          stroke={strokeColor}
          strokeWidth={interactive ? 2 : 1.75}
          strokeLinecap="round"
          strokeLinejoin="round"
          vectorEffect="non-scaling-stroke"
        />

        {/* Price Endpoint Marker */}
        {hoverIdx === null && (
          <circle
            cx={points[points.length - 1].x}
            cy={points[points.length - 1].y}
            r={3.5}
            fill={strokeColor}
            stroke="#FFFFFF"
            strokeWidth={1.5}
          />
        )}

        {/* Price Tag on Right Axis */}
        {interactive && (
          <g>
            <rect
              x={plotWidth}
              y={lastPt.y - 9}
              width={rightGutter}
              height={18}
              fill={isUp ? '#065F46' : '#991B1B'}
              rx={2}
            />
            <text
              x={plotWidth + 6}
              y={lastPt.y + 3.5}
              fill="#FFFFFF"
              fontSize="10"
              fontWeight="bold"
              fontFamily="JetBrains Mono, monospace"
            >
              ${endPrice.toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: endPrice < 1 ? 4 : 1 })}
            </text>
          </g>
        )}

        {/* Interactive Hover Marker & Crosshair */}
        {interactive && hoverIdx !== null && (
          <g>
            <line
              x1={activePoint.x}
              y1={0}
              x2={activePoint.x}
              y2={height}
              stroke="rgba(255, 255, 255, 0.28)"
              strokeWidth={1}
              strokeDasharray="3,3"
              vectorEffect="non-scaling-stroke"
            />
            <line
              x1={0}
              y1={activePoint.y}
              x2={plotWidth}
              y2={activePoint.y}
              stroke="rgba(255, 255, 255, 0.28)"
              strokeWidth={1}
              strokeDasharray="3,3"
              vectorEffect="non-scaling-stroke"
            />
            <circle
              cx={activePoint.x}
              cy={activePoint.y}
              r={4.5}
              fill="#FFFFFF"
              stroke={strokeColor}
              strokeWidth={2}
            />
            <rect
              x={plotWidth}
              y={activePoint.y - 9}
              width={rightGutter}
              height={18}
              fill="#27272A"
              stroke="#71717A"
              strokeWidth={1}
              rx={2}
            />
            <text
              x={plotWidth + 6}
              y={activePoint.y + 3.5}
              fill="#FFFFFF"
              fontSize="10"
              fontFamily="JetBrains Mono, monospace"
            >
              ${activePoint.val.toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: activePoint.val < 1 ? 4 : 1 })}
            </text>
          </g>
        )}
      </svg>
    </div>
  );
}



function InfoBadge({ topic, text, anchor, onNavigate }: { topic: string, text: string, anchor: string, onNavigate: (anchor: string) => void }) {
  return (
    <span className="info-badge-container">
      <span className="info-badge-icon">?</span>
      <span className="info-badge-tooltip">
        <span style={{ display: 'block', marginBottom: '4px', fontWeight: 'bold', color: '#FFFFFF' }}>{topic}</span>
        {text}
        <a className="info-badge-link" onClick={() => onNavigate(anchor)}>
          Learn more in Model School →
        </a>
      </span>
    </span>
  );
}

function SignalIcon({ signal }: { signal: string }) {
  if (signal === 'BUY') return <TrendingUp size={16} />;
  if (signal === 'SELL') return <TrendingDown size={16} />;
  return <Minus size={16} />;
}

function CandlestickChart({ 
  data: rawData, 
  width: propWidth = "100%", 
  height = 120, 
  interactive = true,
  maxCandles,
  isLive,
  candleSource
}: { 
  data: Candle[]; 
  width?: number | string; 
  height?: number; 
  interactive?: boolean;
  maxCandles?: number;
  isLive?: boolean;
  candleSource?: string;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [containerWidth, setContainerWidth] = useState<number>(0);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const el = containerRef.current;
    const update = () => {
      if (el && el.clientWidth > 0) setContainerWidth(el.clientWidth);
    };
    update();
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        if (entry.contentRect.width > 0) {
          setContainerWidth(Math.round(entry.contentRect.width));
        }
      }
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const data = React.useMemo(() => {
    if (!rawData || rawData.length === 0) return [];
    const limit = maxCandles || (interactive ? undefined : 48);
    if (!limit || rawData.length <= limit) return rawData;
    
    const groupSize = Math.ceil(rawData.length / limit);
    const result: Candle[] = [];
    for (let i = 0; i < rawData.length; i += groupSize) {
      const group = rawData.slice(i, i + groupSize);
      if (group.length === 0) continue;
      result.push({
        open: group[0].open,
        high: Math.max(...group.map(c => c.high)),
        low: Math.min(...group.map(c => c.low)),
        close: group[group.length - 1].close,
        time: group[group.length - 1].time,
        timestamp: group[group.length - 1].timestamp
      });
    }
    return result;
  }, [rawData, maxCandles, interactive]);

  const priceTicks = React.useMemo(() => {
    if (!interactive || !data || data.length === 0) return [];
    const highs = data.map(c => c.high);
    const lows = data.map(c => c.low);
    const min = Math.min(...lows);
    const max = Math.max(...highs);
    const r = max - min || 1;
    const padBottom = 12;
    const padTop = 24;
    const pH = height - padTop - padBottom;
    const ticks = [];
    const count = 4;
    for (let i = 0; i <= count; i++) {
      const val = min + (r * i) / count;
      const y = height - padBottom - ((val - min) / r) * pH;
      ticks.push({ val, y });
    }
    return ticks;
  }, [data, interactive, height]);

  if (!rawData || rawData.length === 0 || data.length === 0) {
    return <div ref={containerRef} style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', textAlign: 'center', padding: '1rem' }}>No candlestick data available.</div>;
  }

  const highs = data.map(c => c.high);
  const lows = data.map(c => c.low);

  const minVal = Math.min(...lows);
  const maxVal = Math.max(...highs);
  const range = maxVal - minVal || 1;

  const rightGutter = interactive ? 65 : 0;
  const width = containerWidth > 0 ? containerWidth : 900;
  const plotWidth = Math.max(100, width - rightGutter);

  const paddingTop = interactive ? 24 : 4;
  const paddingBottom = interactive ? 12 : 4;
  const plotHeight = height - paddingTop - paddingBottom;

  const getY = (val: number) => {
    return height - paddingBottom - ((val - minVal) / range) * plotHeight;
  };

  const candleStep = plotWidth / data.length;
  const candleBodyWidth = Math.max(1.8, Math.min(8, candleStep * 0.72));
  const activeCandle = hoverIdx !== null && hoverIdx >= 0 && hoverIdx < data.length ? data[hoverIdx] : data[data.length - 1];

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!interactive) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    if (mouseX < 0 || mouseX > plotWidth) return;
    const idx = Math.min(data.length - 1, Math.max(0, Math.floor(mouseX / candleStep)));
    setHoverIdx(idx);
  };

  const latestCandle = data[data.length - 1];
  const latestY = getY(latestCandle.close);
  const isLatestBullish = latestCandle.close >= latestCandle.open;


  return (
    <div 
      ref={containerRef}
      style={{ 
        width: typeof propWidth === 'number' ? `${propWidth}px` : propWidth, 
        position: 'relative', 
        background: interactive ? 'rgba(10, 10, 14, 0.4)' : 'transparent', 
        borderRadius: '8px',
        boxSizing: 'border-box'
      }}
    >
      {/* OHLC Bar at the top - only when interactive */}
      {interactive && (
        <div style={{ display: 'flex', gap: '0.5rem', fontSize: '0.75rem', fontFamily: "'JetBrains Mono', monospace", color: 'var(--text-secondary)', marginBottom: '0.5rem', justifyContent: 'space-between', borderBottom: '1px solid rgba(255, 255, 255, 0.06)', paddingBottom: '0.3rem' }}>
          <div style={{ display: 'flex', gap: '0.6rem' }}>
            <span>O: <strong style={{ color: '#FFFFFF' }}>${activeCandle.open.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: activeCandle.open < 1 ? 5 : 2 })}</strong></span>
            <span>H: <strong style={{ color: '#10B981' }}>${activeCandle.high.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: activeCandle.high < 1 ? 5 : 2 })}</strong></span>
            <span>L: <strong style={{ color: '#EF4444' }}>${activeCandle.low.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: activeCandle.low < 1 ? 5 : 2 })}</strong></span>
            <span>C: <strong style={{ color: activeCandle.close >= activeCandle.open ? '#10B981' : '#EF4444' }}>${activeCandle.close.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: activeCandle.close < 1 ? 5 : 2 })}</strong></span>
          </div>
          <span style={{ color: '#FFFFFF', fontWeight: 600, display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
            {hoverIdx !== null ? (
              <>
                <span style={{ color: 'var(--text-muted)' }}>Time:</span>
                <span style={{ color: '#FAFAFA' }}>{formatCandleDateTime(activeCandle)}</span>
              </>
            ) : isLive ? (
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', color: '#34D399' }}>
                <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: '#34D399', boxShadow: '0 0 6px #34D399', display: 'inline-block' }}></span>
                Live 5m Stream
              </span>
            ) : (
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', color: '#FBBF24' }}>
                <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: '#FBBF24', display: 'inline-block' }}></span>
                {candleSource === 'offline_historical_replay' ? 'Historical Replay (Offline)' : 'Cached 5m Stream'}
              </span>
            )}
          </span>
        </div>
      )}

      <svg 
        width="100%" 
        height={height} 
        viewBox={`0 0 ${width} ${height}`} 
        style={{ display: 'block', overflow: 'visible', cursor: interactive ? 'crosshair' : 'default' }}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoverIdx(null)}
      >
        {/* Horizontal Gridlines & Price Ticks on Right Axis */}
        {interactive && priceTicks.map((tick, i) => (
          <g key={i}>
            <line
              x1={0}
              y1={tick.y}
              x2={plotWidth}
              y2={tick.y}
              stroke="rgba(255, 255, 255, 0.07)"
              strokeWidth={1}
              strokeDasharray="3,3"
              vectorEffect="non-scaling-stroke"
            />
            <text
              x={plotWidth + 8}
              y={tick.y + 3.5}
              fill="#71717A"
              fontSize="10"
              fontFamily="JetBrains Mono, monospace"
            >
              ${tick.val.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: tick.val < 1 ? 4 : 0 })}
            </text>
          </g>
        ))}

        {/* Right-axis separator line */}
        {interactive && (
          <line
            x1={plotWidth}
            y1={paddingTop}
            x2={plotWidth}
            y2={height - paddingBottom}
            stroke="rgba(255, 255, 255, 0.12)"
            strokeWidth={1}
            vectorEffect="non-scaling-stroke"
          />
        )}

        {/* Price Tag on Right Axis */}
        {interactive && (
          <g>
            <rect
              x={plotWidth}
              y={latestY - 9}
              width={rightGutter}
              height={18}
              fill={isLatestBullish ? '#065F46' : '#991B1B'}
              rx={2}
            />
            <text
              x={plotWidth + 6}
              y={latestY + 3.5}
              fill="#FFFFFF"
              fontSize="10"
              fontWeight="bold"
              fontFamily="JetBrains Mono, monospace"
            >
              ${latestCandle.close.toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: latestCandle.close < 1 ? 4 : 1 })}
            </text>
          </g>
        )}

        {/* Candles */}
        {data.map((candle, idx) => {
          const x = idx * candleStep + candleStep / 2;
          const yOpen = getY(candle.open);
          const yClose = getY(candle.close);
          const yHigh = getY(candle.high);
          const yLow = getY(candle.low);

          const isBullish = candle.close >= candle.open;
          const color = isBullish ? '#10B981' : '#EF4444';
          const top = Math.min(yOpen, yClose);
          const bodyH = Math.max(1.5, Math.abs(yOpen - yClose));

          return (
            <g key={idx}>
              {/* Wick */}
              <line 
                x1={x} 
                y1={yHigh} 
                x2={x} 
                y2={yLow} 
                stroke={color} 
                strokeWidth={1} 
                strokeOpacity={0.8}
                vectorEffect="non-scaling-stroke"
              />
              {/* Body */}
              <rect 
                x={x - candleBodyWidth / 2} 
                y={top} 
                width={candleBodyWidth} 
                height={bodyH} 
                fill={color} 
                rx={0.5}
              />
            </g>
          );
        })}

        {/* Interactive Crosshair (Vertical & Horizontal) */}
        {interactive && hoverIdx !== null && (
          <g>
            {/* Vertical crosshair */}
            <line
              x1={hoverIdx * candleStep + candleStep / 2}
              y1={paddingTop}
              x2={hoverIdx * candleStep + candleStep / 2}
              y2={height - paddingBottom}
              stroke="rgba(255, 255, 255, 0.3)"
              strokeWidth={1}
              strokeDasharray="2,2"
              vectorEffect="non-scaling-stroke"
            />
            {/* Horizontal crosshair to right axis */}
            <line
              x1={0}
              y1={getY(activeCandle.close)}
              x2={plotWidth}
              y2={getY(activeCandle.close)}
              stroke="rgba(255, 255, 255, 0.3)"
              strokeWidth={1}
              strokeDasharray="2,2"
              vectorEffect="non-scaling-stroke"
            />
            {/* Hovered Price Tag on Right Axis */}
            <rect
              x={plotWidth}
              y={getY(activeCandle.close) - 9}
              width={rightGutter}
              height={18}
              fill="#27272A"
              stroke="#71717A"
              strokeWidth={1}
              rx={2}
            />
            <text
              x={plotWidth + 6}
              y={getY(activeCandle.close) + 3.5}
              fill="#FFFFFF"
              fontSize="10"
              fontFamily="JetBrains Mono, monospace"
            >
              ${activeCandle.close.toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: activeCandle.close < 1 ? 4 : 1 })}
            </text>
          </g>
        )}
      </svg>
    </div>
  );
}

interface MonteCarloConeChartProps {
  mcData: MonteCarloData;
}

export function MonteCarloConeChart({ mcData }: MonteCarloConeChartProps) {
  const [hoverStep, setHoverStep] = useState<number | null>(null);

  const width = 600;
  const height = 250;
  const paddingLeft = 50;
  const paddingRight = 20;
  const paddingTop = 15;
  const paddingBottom = 30;

  // Precompute static geometry once per dataset change with useMemo
  const geom = React.useMemo(() => {
    const allVals = [
      ...mcData.percentiles["5"],
      ...mcData.percentiles["25"],
      ...mcData.percentiles["50"],
      ...mcData.percentiles["75"],
      ...mcData.percentiles["95"],
      ...mcData.sample_paths.flat()
    ];
    let yMin = Infinity;
    let yMax = -Infinity;
    for (let i = 0; i < allVals.length; i++) {
      if (allVals[i] < yMin) yMin = allVals[i];
      if (allVals[i] > yMax) yMax = allVals[i];
    }
    yMin *= 0.98;
    yMax *= 1.02;

    const plotWidth = width - paddingLeft - paddingRight;
    const plotHeight = height - paddingTop - paddingBottom;
    const stepCount = mcData.step_count;

    const getX = (t: number) => paddingLeft + (t / (stepCount - 1)) * plotWidth;
    const getY = (v: number) => height - paddingBottom - ((v - yMin) / (yMax - yMin)) * plotHeight;

    const outerPoints = [
      ...mcData.percentiles["5"].map((val: number, t: number) => `${getX(t).toFixed(1)},${getY(val).toFixed(1)}`),
      ...[...mcData.percentiles["95"]].reverse().map((val: number, t: number) => `${getX(stepCount - 1 - t).toFixed(1)},${getY(val).toFixed(1)}`)
    ].join(' ');

    const innerPoints = [
      ...mcData.percentiles["25"].map((val: number, t: number) => `${getX(t).toFixed(1)},${getY(val).toFixed(1)}`),
      ...[...mcData.percentiles["75"]].reverse().map((val: number, t: number) => `${getX(stepCount - 1 - t).toFixed(1)},${getY(val).toFixed(1)}`)
    ].join(' ');

    const medianPoints = mcData.percentiles["50"].map((val: number, t: number) => `${getX(t).toFixed(1)},${getY(val).toFixed(1)}`).join(' ');

    const samplePathsPoints = mcData.sample_paths.map((path: number[]) =>
      path.map((val: number, t: number) => `${getX(t).toFixed(1)},${getY(val).toFixed(1)}`).join(' ')
    );

    const ticksCount = 4;
    const yTicks = Array.from({ length: ticksCount + 1 }, (_, index) => {
      return yMin + (index / ticksCount) * (yMax - yMin);
    });

    const xTicks = [0, Math.floor(stepCount * 0.2), Math.floor(stepCount * 0.4), Math.floor(stepCount * 0.6), Math.floor(stepCount * 0.8), stepCount - 1];

    return { yMin, yMax, getX, getY, outerPoints, innerPoints, medianPoints, samplePathsPoints, yTicks, xTicks, plotWidth };
  }, [mcData]);

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const mouseX = ((e.clientX - rect.left) / rect.width) * width;
    const pct = (mouseX - paddingLeft) / geom.plotWidth;
    const step = Math.min(mcData.step_count - 1, Math.max(0, Math.round(pct * (mcData.step_count - 1))));
    if (step !== hoverStep) {
      setHoverStep(step);
    }
  };

  const handleMouseLeave = () => {
    if (hoverStep !== null) {
      setHoverStep(null);
    }
  };

  return (
    <>
      <svg
        width="100%"
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        style={{ overflow: 'visible', display: 'block', cursor: 'crosshair' }}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      >
        {/* Grid Lines (Horizontal) */}
        {geom.yTicks.map((val, idx) => (
          <g key={idx}>
            <line
              x1={paddingLeft}
              y1={geom.getY(val)}
              x2={width - paddingRight}
              y2={geom.getY(val)}
              stroke="rgba(255,255,255,0.05)"
              strokeWidth="1"
            />
            <text
              x={paddingLeft - 8}
              y={geom.getY(val) + 4}
              fill="var(--text-secondary)"
              fontSize="9"
              textAnchor="end"
            >
              ${Math.round(val).toLocaleString()}
            </text>
          </g>
        ))}

        {/* Grid Lines (Vertical) */}
        {geom.xTicks.map((step, idx) => (
          <g key={idx}>
            <line
              x1={geom.getX(step)}
              y1={paddingTop}
              x2={geom.getX(step)}
              y2={height - paddingBottom}
              stroke="rgba(255,255,255,0.05)"
              strokeWidth="1"
            />
            <text
              x={geom.getX(step)}
              y={height - paddingBottom + 16}
              fill="var(--text-secondary)"
              fontSize="9"
              textAnchor="middle"
            >
              Step {step}
            </text>
          </g>
        ))}

        {/* Percentile Band: 5th-95th (Outer) */}
        <polygon
          points={geom.outerPoints}
          fill="rgba(255, 255, 255, 0.04)"
          stroke="rgba(255, 255, 255, 0.08)"
          strokeWidth="1"
        />

        {/* Percentile Band: 25th-75th (Inner) */}
        <polygon
          points={geom.innerPoints}
          fill="rgba(255, 255, 255, 0.08)"
          stroke="rgba(255, 255, 255, 0.16)"
          strokeWidth="1"
        />

        {/* 5 Sample Simulation Paths */}
        {geom.samplePathsPoints.map((pts: string, idx: number) => (
          <polyline
            key={idx}
            points={pts}
            fill="none"
            stroke="rgba(255, 255, 255, 0.18)"
            strokeWidth="0.8"
          />
        ))}

        {/* Median (50th Percentile) Path */}
        <polyline
          points={geom.medianPoints}
          fill="none"
          stroke="#FFFFFF"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* Interactive Hover Guidance Line */}
        {hoverStep !== null && (
          <g>
            <line
              x1={geom.getX(hoverStep)}
              y1={paddingTop}
              x2={geom.getX(hoverStep)}
              y2={height - paddingBottom}
              stroke="rgba(255, 255, 255, 0.45)"
              strokeWidth="1.2"
              strokeDasharray="4,4"
            />
            {/* Circle marker on Median Path */}
            <circle
              cx={geom.getX(hoverStep)}
              cy={geom.getY(mcData.percentiles["50"][hoverStep])}
              r="5"
              fill="#FFFFFF"
              stroke="#000000"
              strokeWidth="2"
            />
          </g>
        )}
      </svg>

      {/* Hover Interactive Tooltip floating overlay */}
      {hoverStep !== null && (
        <div className="glass-panel" style={{
          position: 'absolute',
          top: '1.5rem',
          left: geom.getX(hoverStep) > width / 2 ? '2rem' : 'auto',
          right: geom.getX(hoverStep) <= width / 2 ? '2rem' : 'auto',
          padding: '0.8rem 1rem',
          fontSize: '0.8rem',
          zIndex: 10,
          width: '180px',
          background: 'rgba(12, 12, 14, 0.97)',
          border: '1px solid rgba(255, 255, 255, 0.2)',
          boxShadow: '0 8px 30px rgba(0, 0, 0, 0.7)',
          borderRadius: '8px',
          pointerEvents: 'none'
        }}>
          <div style={{ fontWeight: 'bold', color: '#FFFFFF', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '0.25rem', marginBottom: '0.4rem' }}>
            Step {hoverStep}
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-secondary)' }}>95th% (Top):</span> <strong>${Math.round(mcData.percentiles["95"][hoverStep]).toLocaleString()}</strong></div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-secondary)' }}>75th% (High):</span> <strong>${Math.round(mcData.percentiles["75"][hoverStep]).toLocaleString()}</strong></div>
            <div style={{ display: 'flex', justifyContent: 'space-between', color: '#FFFFFF' }}><span style={{ fontWeight: 'bold' }}>50th% (Median):</span> <strong>${Math.round(mcData.percentiles["50"][hoverStep]).toLocaleString()}</strong></div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-secondary)' }}>25th% (Low):</span> <strong>${Math.round(mcData.percentiles["25"][hoverStep]).toLocaleString()}</strong></div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}><span style={{ color: 'var(--text-secondary)' }}>5th% (Bottom):</span> <strong>${Math.round(mcData.percentiles["5"][hoverStep]).toLocaleString()}</strong></div>
          </div>
        </div>
      )}
    </>
  );
}

function getPlainSignal(signal?: string): string {
  return ({ Long: 'Long target', Short: 'Short target', Cash: 'Cash target' } as Record<string, string>)[targetLabel(signal)] ?? 'Unavailable';
}


function getPlainRegime(regime?: string): string {
  const r = (regime || '').toLowerCase();
  if ((r.includes('high') && r.includes('bull')) || r.includes('active') || r.includes('parabolic')) return 'Active Bull Rally';
  if (r.includes('bull') || r.includes('uptrend') || r.includes('rally') || r.includes('accumulation')) return 'Calm Uptrend';
  if (r.includes('bear') || r.includes('crash') || r.includes('turbulent') || r.includes('pullback')) return 'Turbulent Pullback';
  if (r.includes('sideways') || r.includes('quiet') || r.includes('range') || r.includes('consolidation') || r.includes('neutral')) return 'Sideways Consolidation';
  return regime || 'Sideways Consolidation';
}

function getCard1Details(regimeRaw?: string, signal?: string) {
  const plainRegime = getPlainRegime(regimeRaw);
  const sig = (signal || '').toUpperCase();
  const isBull = sig === 'BUY';
  const isBear = sig === 'SELL' || sig === 'SHORT';

  // Format allocation and explanation according to canonical 3-action space
  if (isBull) {
    return {
      regimeDisplay: plainRegime,
      explanation: `The classified market regime is ${plainRegime}. The policy targets Long.`,
      allocation: 'Target Long'
    };
  } else if (isBear) {
    return {
      regimeDisplay: plainRegime,
      explanation: `The classified market regime is ${plainRegime}. The policy targets Short.`,
      allocation: 'Target Short'
    };
  } else {
    return {
      regimeDisplay: plainRegime,
      explanation: `The classified market regime is ${plainRegime}. The policy targets Cash.`,
      allocation: 'Target Cash'
    };
  }
}


function App() {
  const [isCopilotCollapsed, setIsCopilotCollapsed] = useState(false);
  const [copilotCenterTab, setCopilotCenterTab] = useState<'dashboard' | 'overview' | 'deep_analytics' | 'school'>('dashboard');
  const [activeTab, setActiveTab] = useState<'dashboard' | 'analytics' | 'deep_analytics' | 'chat' | 'school'>('dashboard');
  const [deepSubTab, setDeepSubTab] = useState<'all' | 'backtest' | 'shap' | 'regime' | 'wfv' | 'montecarlo'>('all');
  const [overviewChartMode, setOverviewChartMode] = useState<'line' | 'candles'>('line');
  const [pendingScrollHash, setPendingScrollHash] = useState<string | null>(null);
  const [isTelemetryCollapsed, setIsTelemetryCollapsed] = useState(false);
  const [viewMode, setViewMode] = useState<'simple' | 'advanced'>('simple');
  const [simpleScenario, setSimpleScenario] = useState<'normal' | 'choppy' | 'crash'>('crash');
  const [showTradeMarkers, setShowTradeMarkers] = useState(false);

  const handleToggleViewMode = (mode: 'simple' | 'advanced') => {
    setViewMode(mode);
  };
  const centerCanvasRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (centerCanvasRef.current) {
      centerCanvasRef.current.scrollTop = 0;
    }
  }, [copilotCenterTab]);

  const handleNavigateSchool = (hash: string) => {
    setActiveTab('school');
    setPendingScrollHash(hash);
  };

  useEffect(() => {
    if (activeTab === 'school' && pendingScrollHash) {
      setTimeout(() => {
        const element = document.getElementById(pendingScrollHash.replace('#', ''));
        if (element) {
          element.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
        setPendingScrollHash(null);
      }, 100);
    }
  }, [activeTab, pendingScrollHash]);
  const [scoutData, setScoutData] = useState<{ Crypto: AssetData[]; ETFs: AssetData[] } | null>(null);
  const [scoutConnectionLost, setScoutConnectionLost] = useState(false);
  const [selectedAsset, setSelectedAsset] = useState<AssetData | null>(null);
  const [chatMessages, setChatMessages] = useState([
    { role: 'assistant', content: 'Choose an asset and I can help you understand its signal, indicators and risks.' }
  ]);
  const [chatInput, setChatInput] = useState('');
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Request ID refs for race condition protection across rapid asset switching
  const analyticsReqIdRef = useRef(0);
  const mcReqIdRef = useRef(0);
  const mcAssetRef = useRef<string | null>(null);

  // New states for real-time model integration
  const [backtestMetrics, setBacktestMetrics] = useState<BacktestMetrics | null>(null);
  const selectedRegime = 'recent';
  const [shapData, setShapData] = useState<ShapData | null>(null);
  const [loadingBacktest, setLoadingBacktest] = useState(false);
  const [loadingShap, setLoadingShap] = useState(false);
  const loadingAnalytics = loadingBacktest || loadingShap;
  const backtestCache = useRef(createRequestCache<BacktestMetrics>(300_000));
  const shapCache = useRef(createRequestCache<ShapData>(60_000));
  const mcCache = useRef(createRequestCache<MonteCarloData>(300_000));

  // Walk-Forward Validation state
  const [wfvData, setWfvData] = useState<WalkForwardRow[] | null>(null);
  const [loadingWFV, setLoadingWFV] = useState(false);

  const fetchWalkForward = async () => {
    setLoadingWFV(true);
    try {
      const res = await fetch('/api/walk-forward');
      if (res.ok) {
        const json = await res.json();
        setWfvData(json.data);
      }
    } catch (err) {
      console.error('Error fetching walk forward data:', err);
    }
    setLoadingWFV(false);
  };


  // Monte Carlo simulation states
  const [mcData, setMcData] = useState<MonteCarloData | null>(null);
  const [loadingMC, setLoadingMC] = useState(false);
  const [mcSimulations, setMcSimulations] = useState(50);
  const [mcNoise, setMcNoise] = useState(0.002);
  const [mcSwan, setMcSwan] = useState(false);

  // Model comparison states
  const [comparisonData, setComparisonData] = useState<ComparisonRow[] | null>(null);
  const [loadingComparison, setLoadingComparison] = useState(false);

  const fetchComparison = async () => {
    setLoadingComparison(true);
    try {
      const res = await fetch('/api/comparison');
      if (res.ok) {
        const data = await res.json();
        setComparisonData(data.results);
      }
    } catch (err) {
      console.error('Error fetching comparison:', err);
    } finally {
      setLoadingComparison(false);
    }
  };

  useEffect(() => {
    let isMounted = true;
    if (activeTab === 'school' && !comparisonData) {
      fetch('/api/comparison')
        .then(res => res.ok ? res.json() : Promise.reject())
        .then(data => {
          if (isMounted) setComparisonData(data.results);
        })
        .catch(err => console.error('Error fetching comparison:', err));
    }
    return () => {
      isMounted = false;
    };
  }, [activeTab, comparisonData]);

  const fetchScoutData = () => {
    fetch('/api/scout')
      .then(res => {
        if (!res.ok) throw new Error('Scout fetch failed');
        return res.json();
      })
      .then(data => {
        setScoutConnectionLost(false);
        setScoutData(data);
        setSelectedAsset(prev => {
          const all = [...(data.Crypto || []), ...(data.ETFs || [])];
          const hash = window.location.hash;
          if (!prev && hash.startsWith('#/asset/')) {
            const assetName = decodeURIComponent(hash.replace('#/asset/', '').split('/')[0]);
            const match = all.find(a => a.label.toLowerCase() === assetName.toLowerCase());
            if (match) return match;
          }
          if (prev) {
            const updated = all.find((a: AssetData) => a.label === prev.label);
            if (updated) {
              return {
                ...prev,
                price: updated.price,
                signal: updated.signal,
                confidence: updated.confidence,
                sentiment: updated.sentiment,
                regime: updated.regime,
                candles: updated.candles,
                history: updated.history,
                summary: updated.summary,
                probs: updated.probs,
                action: updated.action,
                model_eval_time: updated.model_eval_time,
                quote_time: updated.quote_time,
                candle_source: updated.candle_source,
                is_synthetic_candles: updated.is_synthetic_candles,
                quote_is_live: updated.quote_is_live,
                candle_is_live: updated.candle_is_live,
                last_candle_time: updated.last_candle_time
              };
            }
            return prev;
          }
          return null;
        });
      })
      .catch(err => {
        console.error('API Error:', err);
        setScoutConnectionLost(true);
      });
  };

  useEffect(() => {
    fetchScoutData();
    const interval = setInterval(fetchScoutData, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchAnalytics = async (assetName: string, regime = selectedRegime, refresh = false) => {
    const reqId = ++analyticsReqIdRef.current;
    const backtestUrl = `/api/backtest?asset=${encodeURIComponent(assetName)}&regime=${encodeURIComponent(regime)}`;
    const shapUrl = `/api/shap?asset=${encodeURIComponent(assetName)}`;
    const savedBacktest = backtestCache.current.peek(backtestUrl);
    const savedShap = shapCache.current.peek(shapUrl);
    setBacktestMetrics(savedBacktest ?? null);
    setShapData(savedShap ?? null);
    setLoadingBacktest(refresh || !savedBacktest);
    setLoadingShap(refresh || !savedShap);
    // Publish each result as soon as it arrives; the slower panel never hides it.
    await Promise.allSettled([
      backtestCache.current.get(backtestUrl, () => fetchJson<BacktestMetrics>(backtestUrl), refresh)
        .then(data => { if (reqId === analyticsReqIdRef.current) setBacktestMetrics(data); })
        .catch(err => console.error('Backtest request failed:', err))
        .finally(() => { if (reqId === analyticsReqIdRef.current) setLoadingBacktest(false); }),
      shapCache.current.get(shapUrl, () => fetchJson<ShapData>(shapUrl), refresh)
        .then(data => { if (reqId === analyticsReqIdRef.current) setShapData(data); })
        .catch(err => console.error('SHAP request failed:', err))
        .finally(() => { if (reqId === analyticsReqIdRef.current) setLoadingShap(false); }),
    ]);
  };

  const mcUrl = (assetName: string, sims: number, noise: number, swan: boolean) =>
    `/api/montecarlo?asset=${encodeURIComponent(assetName)}&simulations=${sims}&noise_std=${noise}&inject_swan=${swan}`;

  const selectMonteCarlo = (assetName: string) => {
    if (mcAssetRef.current === assetName) return;
    mcAssetRef.current = assetName;
    ++mcReqIdRef.current; // An old asset's response must not overwrite the selection.
    setLoadingMC(false);
    setMcData(mcCache.current.peek(mcUrl(assetName, mcSimulations, mcNoise, mcSwan)) ?? null);
  };

  const fetchMonteCarlo = async (assetName: string, sims = mcSimulations, noise = mcNoise, swan = mcSwan) => {
    mcAssetRef.current = assetName;
    const reqId = ++mcReqIdRef.current;
    const url = mcUrl(assetName, sims, noise, swan);
    setLoadingMC(true);
    setMcData(null);
    try {
      const data = await mcCache.current.get(url, () => fetchJson<MonteCarloData>(url));
      if (reqId === mcReqIdRef.current) setMcData(data);
    } catch (err) {
      console.error('Error fetching Monte Carlo:', err);
    } finally {
      if (reqId === mcReqIdRef.current) setLoadingMC(false);
    }
  };

  const handleOpenDeepAnalytics = (asset?: AssetData) => {
    const target = asset || selectedAsset || (allAssets.length > 0 ? allAssets[0] : null);
    if (target) {
      setSelectedAsset(target);
    }
    setActiveTab('deep_analytics');
    setCopilotCenterTab('deep_analytics');
  };

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages, loading]);

  const sendChat = async (message: string, overrideAsset?: AssetData | null, displayMessage?: string) => {
    if (!message.trim()) return;
    const userText = displayMessage || message;
    setChatMessages(prev => [...prev, { role: 'user', content: userText }]);
    setChatInput('');
    setLoading(true);
    try {
      const payload: ChatPayload = {
        message,
        history: chatMessages.slice(-4).map(msg => ({ role: msg.role, content: msg.content }))
      };
      const activeAsset = overrideAsset !== undefined 
        ? (overrideAsset || undefined) 
        : (copilotCenterTab === 'dashboard' ? undefined : selectedAsset);

      if (activeAsset) {
        const freshList = scoutData ? [...scoutData.Crypto, ...scoutData.ETFs] : [];
        const fresh = freshList.find(a => a.label.toLowerCase() === activeAsset.label.toLowerCase()) || activeAsset;
        payload.asset_name = fresh.label;
        payload.price = fresh.price;
        payload.signal = fresh.signal;
        payload.confidence = fresh.confidence;
        payload.sentiment = fresh.sentiment;
      }
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `Server status ${res.status}`);
      }
      const data = await res.json();
      if (data && data.response) {
        setChatMessages(prev => [...prev, { role: 'assistant', content: data.response }]);
      } else {
        throw new Error("Empty response from advisor");
      }
    } catch (err) {
      console.error("Chat error:", err);
      setChatMessages(prev => [...prev, { role: 'assistant', content: 'Sorry, I am currently having trouble reaching the advisor backend. Please try again in a moment.' }]);
    }
    setLoading(false);
  };

  const handleSendChat = (e: React.FormEvent) => {
    e.preventDefault();
    sendChat(chatInput);
  };

  const getSparkColor = (signal: string) => {
    if (signal === 'BUY') return '#34D399';
    if (signal === 'SELL') return '#F87171';
    return '#A1A1AA';
  };

  const allAssets = scoutData ? [...scoutData.Crypto, ...scoutData.ETFs] : [];
  const feedAssets = selectedAsset ? [selectedAsset] : allAssets;
  const liveQuoteCount = feedAssets.filter(asset => asset.quote_is_live === true).length;
  const feedStatus = scoutConnectionLost ? 'Connection lost — last received data'
    : liveQuoteCount === feedAssets.length && feedAssets.length > 0 ? 'LIVE (SYNCED)'
    : liveQuoteCount > 0 ? `MIXED (${liveQuoteCount}/${feedAssets.length} LIVE)` : 'OFFLINE (CACHE / ARCHIVE)';
  const feedStatusColor = scoutConnectionLost ? '#F87171'
    : liveQuoteCount === feedAssets.length && feedAssets.length > 0 ? '#10B981' : '#F59E0B';

  const handleCopilotAssetSelect = (asset: AssetData) => {
    setSelectedAsset(asset);
    setViewMode('simple');
    if (copilotCenterTab === 'dashboard') {
      setCopilotCenterTab('overview');
    }
  };

  // Data requests follow the selected asset. Event handlers only update UI state.
  useEffect(() => {
    if (!selectedAsset) return;
    const assetLabel = selectedAsset.label;
    const timer = window.setTimeout(() => {
      fetchAnalytics(assetLabel);
      if (!wfvData) fetchWalkForward();
      selectMonteCarlo(assetLabel);
    }, 0);
    // Requests are keyed by asset; cache and request IDs handle later refreshes.
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedAsset?.label]);

  const handleDiveDeeperWithAI = () => {
    if (!selectedAsset) return;
    setIsCopilotCollapsed(false);
    sendChat(
      `Briefly analyze ${selectedAsset.label}: explain the saved ${getPlainSignal(selectedAsset.signal)}, loaded market phase, data timestamps, and key risk factor in 2-3 sentences.`,
      selectedAsset, 
      `✨ Analyze ${selectedAsset.label}`
    );
  };

  // ─── BROWSER BACK & FORWARD BUTTON HISTORY SYNC ───
  const isPopstateNav = useRef(false);
  const allAssetsRef = useRef<AssetData[]>([]);

  useEffect(() => {
    allAssetsRef.current = scoutData ? [...scoutData.Crypto, ...scoutData.ETFs] : [];
  }, [scoutData]);

  useEffect(() => {
    const handlePopState = () => {
      isPopstateNav.current = true;
      const hash = window.location.hash || '#/dashboard';

      if (hash.startsWith('#/school')) {
        setCopilotCenterTab('school');
        setActiveTab('school');
        const match = hash.match(/#\/school(#.+)/);
        if (match && match[1]) {
          setPendingScrollHash(match[1]);
        }
      } else if (hash.startsWith('#/asset/')) {
        const rawPath = hash.replace('#/asset/', '');
        const segments = rawPath.split('/');
        const assetName = decodeURIComponent(segments[0]);
        const isDeep = segments[1] === 'deep';
        const sub = (segments[2] as 'all' | 'backtest' | 'shap' | 'regime' | 'wfv' | 'montecarlo') || 'all';

        const found = allAssetsRef.current.find(a => a.label.toLowerCase() === assetName.toLowerCase());
        if (found) {
          setSelectedAsset(found);
        }

        if (isDeep) {
          setCopilotCenterTab('deep_analytics');
          setActiveTab('deep_analytics');
          setDeepSubTab(sub);
        } else {
          setCopilotCenterTab('overview');
          setActiveTab('analytics');
        }
      } else {
        setCopilotCenterTab('dashboard');
        setActiveTab('dashboard');
      }
    };

    window.addEventListener('popstate', handlePopState);

    if (window.location.hash && window.location.hash !== '#/dashboard' && window.location.hash !== '#/') {
      handlePopState();
    } else {
      window.history.replaceState({ tab: 'dashboard' }, '', '#/dashboard');
    }

    return () => {
      window.removeEventListener('popstate', handlePopState);
    };
  }, []);

  useEffect(() => {
    if (isPopstateNav.current) {
      isPopstateNav.current = false;
      return;
    }

    let targetHash = '#/dashboard';
    if (copilotCenterTab === 'school') {
      targetHash = '#/school';
    } else if (copilotCenterTab === 'deep_analytics' && selectedAsset) {
      targetHash = `#/asset/${encodeURIComponent(selectedAsset.label)}/deep/${deepSubTab}`;
    } else if (copilotCenterTab === 'overview' && selectedAsset) {
      targetHash = `#/asset/${encodeURIComponent(selectedAsset.label)}`;
    }

    if (window.location.hash !== targetHash) {
      window.history.pushState(
        { tab: copilotCenterTab, asset: selectedAsset?.label, subTab: deepSubTab },
        '',
        targetHash
      );
    }
  }, [copilotCenterTab, selectedAsset, deepSubTab]);

  // ─── TRI-PANE COPILOT RENDERERS ───
  const renderLeftRail = () => {
    return (
    <aside className="tripane-left-rail">
      {/* Crypto Watchlist */}
      <div>
        <div className="watchlist-group-title">
          <span>Crypto Assets</span>
          <span style={{ fontFamily: 'monospace' }}>{scoutData?.Crypto?.length || 0}</span>
        </div>
        {scoutData?.Crypto?.map((asset: AssetData) => {
          const isSelected = selectedAsset?.label === asset.label;
          const trend = getAssetDayTrend(asset);
          return (
            <div
              key={asset.label}
              className={`watchlist-card ${isSelected ? 'active' : ''}`}
              onClick={() => handleCopilotAssetSelect(asset)}
            >
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.15rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <span style={{ fontWeight: 700, fontSize: '0.9rem', color: isSelected ? '#FFFFFF' : 'var(--text-primary)' }}>
                    {asset.label}
                  </span>

                </div>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', justifyContent: 'center' }}>
                <span style={{ fontSize: '0.82rem', fontWeight: 600, fontFamily: 'JetBrains Mono, monospace', color: '#FFFFFF' }}>
                  ${asset.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: asset.price < 1 ? 5 : 2 })}
                </span>
                <span style={{ fontSize: '0.72rem', fontWeight: 600, color: trend.color, fontFamily: 'JetBrains Mono, monospace' }}>
                  {trend.isUp ? '+' : ''}{trend.changePct.toFixed(2)}%
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* ETFs Watchlist */}
      <div>
        <div className="watchlist-group-title">
          <span>ETFs & Equities</span>
          <span style={{ fontFamily: 'monospace' }}>{scoutData?.ETFs?.length || 0}</span>
        </div>
        {scoutData?.ETFs?.map((asset: AssetData) => {
          const isSelected = selectedAsset?.label === asset.label;
          const trend = getAssetDayTrend(asset);
          return (
            <div
              key={asset.label}
              className={`watchlist-card ${isSelected ? 'active' : ''}`}
              onClick={() => handleCopilotAssetSelect(asset)}
            >
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.15rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <span style={{ fontWeight: 700, fontSize: '0.9rem', color: isSelected ? '#FFFFFF' : 'var(--text-primary)' }}>
                    {asset.label}
                  </span>

                </div>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', justifyContent: 'center' }}>
                <span style={{ fontSize: '0.82rem', fontWeight: 600, fontFamily: 'JetBrains Mono, monospace', color: '#FFFFFF' }}>
                  ${asset.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </span>
                <span style={{ fontSize: '0.72rem', fontWeight: 600, color: trend.color, fontFamily: 'JetBrains Mono, monospace' }}>
                  {trend.isUp ? '+' : ''}{trend.changePct.toFixed(2)}%
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Grounding Disclaimer */}
      <div style={{ marginTop: 'auto', padding: '0.6rem 0.4rem', borderTop: '1px solid rgba(255,255,255,0.06)' }}>
        <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', lineHeight: 1.4 }}>
          🛡️ AI signals powered by reinforcement learning and historical market patterns.
        </div>
      </div>
    </aside>
    );
  };

  const renderSynthesisCard = () => {
    const isDashboard = copilotCenterTab === 'dashboard';

    if (isDashboard) {
      return (
        <div className="synthesis-card">
          <div className="synthesis-header">
            <div className="synthesis-title-row">
              <Sparkles size={17} color="#FFFFFF" />
              <span>Your Market Briefing</span>
            </div>
          </div>

          <p style={{ margin: 0, fontSize: '0.88rem', color: '#E4E4E7', lineHeight: 1.55 }}>
            {summarizeMarket(allAssets).split('\n\n').map((paragraph, index) => (
              <span key={index} style={{ display: 'block', marginTop: index ? '0.65rem' : 0 }}>{paragraph}</span>
            ))}
          </p>
        </div>
      );
    }

    return (
      <div className="synthesis-card">
        <div className="synthesis-header">
          <div className="synthesis-title-row">
            <Sparkles size={17} color="#FFFFFF" />
            <span>{selectedAsset ? `${selectedAsset.label} · Market Summary` : 'Market Summary'}</span>
          </div>
        </div>

        <p style={{ margin: 0, fontSize: '0.92rem', color: '#E8E8EB', lineHeight: 1.6 }}>
          {selectedAsset ? summarizeAsset(selectedAsset) : 'Select an asset to view its available data.'}
        </p>
      </div>
    );
  };

  const renderRightCopilotRail = () => (
    <aside className={`tripane-right-copilot ${isCopilotCollapsed ? 'collapsed' : ''}`}>
      <div className="copilot-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Bot size={18} color="#FFFFFF" />
          <span style={{ fontWeight: 700, fontSize: '0.88rem', color: '#FFFFFF' }}>AI Advisor</span>
          <span style={{ fontSize: '0.65rem', background: 'rgba(255,255,255,0.08)', padding: '0.15rem 0.45rem', borderRadius: '4px', color: '#A1A1AA' }}>
            GEMINI
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <button
            onClick={() => setChatMessages([{ role: 'assistant', content: 'Choose an asset and I can help you understand its signal, indicators and risks.' }])}
            title="Reset Chat"
            style={{ background: 'none', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer', padding: '0.2rem', display: 'flex' }}
          >
            <RotateCcw size={15} />
          </button>
          <button
            onClick={() => setIsCopilotCollapsed(true)}
            title="Collapse Copilot Panel"
            style={{ background: 'none', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer', padding: '0.2rem', display: 'flex' }}
          >
            <ChevronRight size={18} />
          </button>
        </div>
      </div>

      <div style={{ padding: '0.6rem 0.85rem', background: 'rgba(0,0,0,0.25)', borderBottom: '1px solid var(--glass-border)', fontSize: '0.74rem', color: 'var(--text-secondary)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        {!selectedAsset || copilotCenterTab === 'dashboard' ? (
          <>
            <span>Context: <strong style={{ color: '#FFFFFF' }}>Market Overview</strong></span>
            <span style={{ color: 'var(--text-secondary)' }}>{allAssets.length} assets in watchlist</span>
          </>
        ) : (
          <>
            <span>Context: <strong style={{ color: '#FFFFFF' }}>{selectedAsset.label}</strong></span>
            <span>${selectedAsset.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: selectedAsset.price < 1 ? 5 : 2 })} · <strong style={{ color: getSparkColor(selectedAsset.signal) }}>{getPlainSignal(selectedAsset.signal)}</strong></span>
          </>
        )}
      </div>

      <div className="chat-thread" role="log" aria-label="Advisor conversation" aria-live="polite" aria-relevant="additions">
        {chatMessages.map((msg, i) => (
          <div key={i} className={`advisor-message ${msg.role}`}>
            <div className={`chat-avatar ${msg.role}`} style={{ width: '30px', height: '30px' }}>
              {msg.role === 'assistant' ? <Bot size={16} color="white" /> : <div style={{ color: 'white', fontWeight: 'bold', fontSize: '0.75rem' }}>U</div>}
            </div>
            <div className="advisor-message-body">
              <div className="advisor-message-label">{msg.role === 'assistant' ? 'ADVISOR' : 'YOU'}</div>
              <div className="advisor-message-copy" dangerouslySetInnerHTML={{ __html: renderMarkdown(msg.content) }} />
            </div>
          </div>
        ))}
        {loading && (
          <div className="glass-panel chat-bubble" style={{ padding: '0.85rem 1rem', gap: '0.75rem' }}>
            <div className="chat-avatar assistant" style={{ width: '30px', height: '30px' }}><Bot size={16} color="white" /></div>
            <div className="typing-indicator"><span></span><span></span><span></span></div>
          </div>
        )}
        {chatMessages.length === 1 && selectedAsset && copilotCenterTab !== 'dashboard' && (
          <div className="chat-starters">
            <span>A good place to start</span>
            <>
                <button disabled={loading} onClick={() => sendChat('Why this signal? Explain only what the available evidence supports.', selectedAsset, 'Why this signal?')}>Why this signal? <ArrowRight size={14} /></button>
                <button disabled={loading} onClick={() => sendChat('What are the risks of this position?', selectedAsset, 'What are the risks?')}>What are the risks? <ArrowRight size={14} /></button>
                <button disabled={loading} onClick={() => sendChat('What does model confidence mean? Is it prediction accuracy?', selectedAsset, 'What does confidence mean?')}>What does confidence mean? <ArrowRight size={14} /></button>
              </>
          </div>
        )}
        <div ref={chatEndRef} />
      </div>

      <div style={{ padding: '0.75rem 0.85rem', borderTop: '1px solid var(--glass-border)', display: 'flex', flexDirection: 'column', gap: '0.5rem', background: 'rgba(10,10,14,0.4)' }}>
        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>
          Quick Analysis & Suggested Actions
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem' }}>
          {!selectedAsset || copilotCenterTab === 'dashboard' ? (
            <>
              <button className="copilot-action-pill" onClick={() => sendChat("Summarize the current market status, trends, and risk stance across all monitored assets in plain terms.", null, "📊 Market Summary")} style={{ background: 'rgba(255, 255, 255, 0.08)', color: '#FFFFFF', borderColor: 'rgba(255, 255, 255, 0.2)' }}>
                <Sparkles size={11} color="#34D399" /> Market Summary
              </button>
              <button className="copilot-action-pill" onClick={() => sendChat("Compare the risk and stability between stock index ETFs (SPY, QQQ) and crypto (BTC, ETH, DOGE) using the loaded data and its timestamps.", null, "⚖️ Compare Assets")}>
                Compare Assets
              </button>
              <button className="copilot-action-pill" onClick={() => sendChat("Which monitored asset has the strongest signal in the loaded data? Include the data timestamps and limitations.", null, "🎯 Top Opportunities")}>
                Top Opportunities
              </button>
              <button className="copilot-action-pill" onClick={() => sendChat("Give me a brief overview of current market volatility and risk across all assets.", null, "🛡️ Risk Overview")}>
                Risk Overview
              </button>
            </>
          ) : (
            <>
              <button className="copilot-action-pill" onClick={() => handleDiveDeeperWithAI()} style={{ background: 'rgba(255, 255, 255, 0.08)', color: '#FFFFFF', borderColor: 'rgba(255, 255, 255, 0.2)' }}>
                <Sparkles size={11} color="#34D399" /> Deep Dive Analysis
              </button>
              <button className="copilot-action-pill" onClick={() => sendChat(`Why did the saved model snapshot target ${getPlainSignal(selectedAsset.signal)} for ${selectedAsset.label}? Include its evaluation timestamp and explain in plain terms.`, selectedAsset, `🎯 Why ${getPlainSignal(selectedAsset.signal)}?`)}>
                Why this signal?
              </button>
              <button className="copilot-action-pill" onClick={() => sendChat(`Explain the key factors driving ${selectedAsset.label}'s price in plain language.`, selectedAsset, `🔍 Key Price Drivers`)}>
                Key Price Drivers
              </button>
              <button className="copilot-action-pill" onClick={() => {
                fetchMonteCarlo(selectedAsset.label);
                sendChat(`Run a stress test simulation for ${selectedAsset.label} and explain the downside risk in everyday terms.`, selectedAsset, `📉 Market Stress-Test`);
              }}>
                Market Stress-Test
              </button>
              <button className="copilot-action-pill" onClick={() => {
                fetchWalkForward();
                sendChat(`How reliable and consistent has the trading strategy been in testing for ${selectedAsset.label}?`, selectedAsset, `🛡️ Model Reliability`);
              }}>
                Model Reliability
              </button>
            </>
          )}
        </div>
      </div>

      <div style={{ padding: '0.75rem 0.85rem', borderTop: '1px solid var(--glass-border)', background: 'rgba(10,10,14,0.8)' }}>
        <form className="chat-input-container" onSubmit={handleSendChat} style={{ margin: 0 }}>
          <input
            type="text"
            className="chat-input"
            style={{ padding: '0.65rem 2.5rem 0.65rem 0.85rem', fontSize: '0.84rem' }}
            placeholder="Ask about the signal or its risks..."
            aria-label="Message the advisor"
            value={chatInput}
            onChange={e => setChatInput(e.target.value)}
          />
          <button type="submit" aria-label="Send message" disabled={loading || !chatInput.trim()} style={{ position: 'absolute', right: '0.6rem', top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', color: '#FFFFFF', cursor: 'pointer', display: 'flex' }}>
            <Send size={16} />
          </button>
        </form>
      </div>
    </aside>
  );

  const renderFloatingExpandBtn = () => (
    <button className="copilot-floating-expand-btn" onClick={() => setIsCopilotCollapsed(false)}>
      <Bot size={16} />
      <span>AI Advisor</span>
      <ChevronLeft size={16} />
    </button>
  );

  // ─── CORE VIEW RENDERERS ───
  const renderDashboardContent = () => (
              <>
                {/* TOP ASSET SELECTION PANEL */}
                <div className="glass-panel" style={{ padding: '1.5rem', marginBottom: '1.5rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
                    <div>
                      <h2 style={{ margin: 0, fontSize: '1.35rem' }}>Explore Investments</h2>
                      <p style={{ color: 'var(--text-secondary)', margin: '0.3rem 0 0 0', fontSize: '0.9rem' }}>
                        Review AI recommendations, explore performance and understand the risks.
                      </p>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                      <span style={{ fontSize: '0.72rem', background: 'rgba(255, 255, 255, 0.08)', padding: '0.2rem 0.55rem', borderRadius: '4px', color: '#E4E4E7', border: '1px solid var(--glass-border)', fontWeight: 500 }}>
                        ⏱ 30min
                      </span>
                      <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontFamily: 'monospace' }}>
                        5 ASSETS MONITORED
                      </span>
                    </div>
                  </div>

                  {scoutData ? (
                    <>
                      {/* Quick Asset Selector Cards Grid */}
                      <div className="asset-grid" style={{ marginTop: '1.2rem' }}>
                        {allAssets.map(asset => {
                          const isSelected = selectedAsset?.label === asset.label;
                          const trend = getAssetDayTrend(asset);
                          return (
                            <div
                              key={asset.label}
                              className={`glass-panel asset-card ${isSelected ? 'selected' : ''}`}
                              onClick={() => {
                                handleCopilotAssetSelect(asset);
                                setActiveTab('analytics');
                                setCopilotCenterTab('overview');
                              }}
                              style={{ 
                                cursor: 'pointer',
                                border: isSelected ? '1px solid #FFFFFF' : '1px solid var(--glass-border)',
                                transform: isSelected ? 'scale(1.02)' : 'none',
                                transition: 'all 0.2s ease'
                              }}
                            >
                              <div className="asset-card-info">
                                <div style={{ fontWeight: 'bold', fontSize: '1.05rem', color: isSelected ? '#FFFFFF' : 'var(--text-primary)', textAlign: 'left' }}>
                                  {asset.label}
                                </div>
                                <div style={{ fontSize: '1.12rem', fontWeight: 'bold', color: 'white', textAlign: 'left', marginTop: '0.2rem', display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                                  ${asset.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: asset.price < 1 ? 5 : 2 })}
                                  <span style={{ 
                                    fontSize: '0.72rem', 
                                    fontWeight: 600, 
                                    color: trend.color, 
                                    background: trend.isUp ? 'rgba(52, 211, 153, 0.12)' : 'rgba(248, 113, 113, 0.12)', 
                                    border: `1px solid ${trend.isUp ? 'rgba(52, 211, 153, 0.25)' : 'rgba(248, 113, 113, 0.25)'}`,
                                    padding: '0.08rem 0.35rem', 
                                    borderRadius: '4px' 
                                  }}>
                                    {trend.isUp ? '+' : ''}{trend.changePct.toFixed(2)}%
                                  </span>
                                </div>
                                <div style={{ marginTop: '0.45rem' }}>
                                    <span style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginRight: '0.4rem' }}>Model snapshot</span>
                                  <span className={`signal-badge signal-${asset.signal.toLowerCase()}`} style={{ padding: '0.18rem 0.6rem', fontSize: '0.72rem' }}>
                                    <SignalIcon signal={asset.signal} />&nbsp;{getPlainSignal(asset.signal)}
                                  </span>
                                </div>
                              </div>
                              <div className="asset-card-chart">
                                <AreaLineChart candles={asset.candles} data={asset.history} color={trend.color} height={62} interactive={false} />
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </>
                  ) : (
                    <div style={{ padding: '3rem 2rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
                      <div className="skeleton" style={{ width: '100%', height: '60px', marginBottom: '1.5rem', opacity: 0.3 }}></div>
                      <p style={{ fontSize: '1.05rem', color: 'var(--text-secondary)' }}>Loading assets...</p>
                    </div>
                  )}
                </div>

                {/* 2. COMPACT CORNER SYSTEM TELEMETRY DOCK */}
                <div style={{
                  display: 'flex',
                  justifyContent: 'flex-end',
                  marginTop: 'auto',
                  paddingTop: '1rem',
                  paddingBottom: '0.5rem'
                }}>
                  {isTelemetryCollapsed ? (
                    <div
                      className="glass-panel"
                      onClick={() => setIsTelemetryCollapsed(false)}
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '0.65rem',
                        padding: '0.35rem 0.85rem',
                        borderRadius: '20px',
                        cursor: 'pointer',
                        background: 'rgba(18, 18, 20, 0.85)',
                        border: '1px solid var(--glass-border)',
                        boxShadow: '0 4px 16px rgba(0,0,0,0.3)',
                        transition: 'all 0.2s ease'
                      }}
                      title="Click to expand system status"
                    >
                      <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: '#10B981', boxShadow: '0 0 6px #10B981' }}></span>
                      <span style={{ fontSize: '0.72rem', fontWeight: 600, color: '#E4E4E7' }}>System Telemetry</span>
                      <span style={{ fontSize: '0.65rem', color: '#10B981', background: 'rgba(16, 185, 129, 0.12)', padding: '1px 5px', borderRadius: '3px' }}>ONLINE</span>
                      <ChevronUp size={13} color="var(--text-muted)" />
                    </div>
                  ) : (
                    <div
                      className="glass-panel"
                      style={{
                        width: '100%',
                        maxWidth: '340px',
                        padding: '0.75rem 0.9rem',
                        borderRadius: '12px',
                        background: 'rgba(14, 14, 16, 0.88)',
                        border: '1px solid var(--glass-border)',
                        boxShadow: '0 8px 28px rgba(0, 0, 0, 0.45)',
                        transition: 'all 0.2s ease'
                      }}
                    >
                      {/* Telemetry Header */}
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.55rem', borderBottom: '1px solid rgba(255, 255, 255, 0.08)', paddingBottom: '0.4rem' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                          <Radio size={12} color="#10B981" />
                          <span style={{ fontSize: '0.72rem', fontWeight: 700, color: '#E4E4E7', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                            System Status
                          </span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                          <span style={{ fontSize: '0.65rem', color: '#10B981', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '4px' }}>
                            <span style={{ width: '5px', height: '5px', borderRadius: '50%', background: '#10B981', display: 'inline-block' }}></span>
                            APP ONLINE
                          </span>
                          <button
                            onClick={() => setIsTelemetryCollapsed(true)}
                            style={{
                              background: 'transparent',
                              border: 'none',
                              color: 'var(--text-muted)',
                              cursor: 'pointer',
                              padding: '2px',
                              display: 'flex',
                              alignItems: 'center',
                              borderRadius: '4px'
                            }}
                            title="Minimize to corner pill"
                          >
                            <ChevronDown size={13} />
                          </button>
                        </div>
                      </div>

                      {/* 3 Compact Status Sub-Cards */}
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.45rem' }}>
                        {/* Card 1: Trading System Engine */}
                        <div style={{
                          padding: '0.45rem 0.6rem',
                          background: 'rgba(255, 255, 255, 0.02)',
                          borderRadius: '7px',
                          border: '1px solid rgba(255, 255, 255, 0.05)'
                        }}>
                          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.15rem' }}>
                            <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Trading System Engine</span>
                            <span style={{ fontSize: '0.65rem', color: '#10B981', fontWeight: 700 }}>ONLINE</span>
                          </div>
                          <div style={{ fontSize: '0.8rem', fontWeight: 700, color: '#FFFFFF' }}>PPO Transformer Architecture</div>
                          <p style={{ fontSize: '0.68rem', color: 'var(--text-muted)', margin: '0.2rem 0 0 0', lineHeight: 1.3 }}>
                            Actor-Critic model trained on continuous tensors & FinBERT.
                          </p>
                        </div>

                        {/* Card 2: Market Data Feeds */}
                        <div style={{
                          padding: '0.45rem 0.6rem',
                          background: 'rgba(255, 255, 255, 0.02)',
                          borderRadius: '7px',
                          border: '1px solid rgba(255, 255, 255, 0.05)'
                        }}>
                          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.15rem' }}>
                            <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Market Data Feeds</span>
                            <span style={{
                              fontSize: '0.65rem',
                              color: feedStatusColor,
                              fontWeight: 700
                            }}>
                              {feedStatus}
                            </span>
                          </div>
                          <div style={{ fontSize: '0.8rem', fontWeight: 700, color: '#FFFFFF' }}>Binance + Yahoo Finance</div>
                          <p style={{ fontSize: '0.68rem', color: 'var(--text-muted)', margin: '0.2rem 0 0 0', lineHeight: 1.3 }}>
                            5m candles may be live, cached or historical; volatility estimates use loaded data.
                          </p>
                        </div>

                        {/* Card 3: Deep Analytics */}
                        <div 
                          onClick={() => handleOpenDeepAnalytics()}
                          style={{
                            padding: '0.45rem 0.6rem',
                            background: 'rgba(255, 255, 255, 0.02)',
                            borderRadius: '7px',
                            border: '1px solid rgba(255, 255, 255, 0.05)',
                            cursor: 'pointer',
                            transition: 'border-color 0.15s ease'
                          }}
                          title="Click to view deep analytics"
                        >
                          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.15rem' }}>
                            <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Deep Analytics</span>
                            <span style={{ fontSize: '0.65rem', color: '#A1A1AA', fontWeight: 700 }}>READY</span>
                          </div>
                          <div style={{ fontSize: '0.8rem', fontWeight: 700, color: '#FFFFFF', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                            <span>Dedicated Page Available</span>
                            <span style={{ fontSize: '0.68rem', color: 'var(--text-secondary)' }}>→</span>
                          </div>
                          <p style={{ fontSize: '0.68rem', color: 'var(--text-muted)', margin: '0.2rem 0 0 0', lineHeight: 1.3 }}>
                            Click to inspect full backtests, market regimes, and key decision drivers.
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </>

    );

  const renderAnalyticsContent = () => (
              <>
                {/* DEEP ANALYTICS & CHARTS PANEL */}
                {selectedAsset ? (
                  <div className="glass-panel" style={{ padding: '2rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
                      <div>
                        <h2 style={{ margin: 0, fontSize: '1.6rem' }}>{selectedAsset.label}</h2>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap', fontSize: '0.82rem', color: 'var(--text-secondary)', marginTop: '0.35rem' }}>
                          <span style={{ color: '#FAFAFA', fontWeight: 600 }}>
                            {scoutData?.Crypto.some(c => c.label === selectedAsset.label) || selectedAsset.label.includes('BTC') || selectedAsset.label.includes('ETH') || selectedAsset.label.includes('DOGE') ? 'Binance Spot' : 'Yahoo Finance'}
                          </span>
                          <span style={{ opacity: 0.4 }}>•</span>
                          <span>
                            {scoutData?.Crypto.some(c => c.label === selectedAsset.label) || selectedAsset.label.includes('BTC') || selectedAsset.label.includes('ETH') || selectedAsset.label.includes('DOGE') ? 'Crypto' : 'Equity ETF'}
                          </span>
                          {scoutConnectionLost ? (
                            <span style={{ display: 'inline-flex', alignItems: 'center', gap: '5px', color: '#F87171' }}>
                              <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: '#F87171', display: 'inline-block' }}></span>
                              Connection lost — last received data
                            </span>
                          ) : (
                            <>
                              <span style={{ display: 'inline-flex', alignItems: 'center', gap: '5px' }}>
                                <span style={{
                                  width: '6px',
                                  height: '6px',
                                  borderRadius: '50%',
                                  background: selectedAsset.quote_is_live ? '#34D399' : '#F59E0B',
                                  boxShadow: selectedAsset.quote_is_live ? '0 0 6px rgba(52, 211, 153, 0.7)' : 'none',
                                  display: 'inline-block'
                                }}></span>
                                {selectedAsset.quote_is_live ? `Live Quote (${selectedAsset.quote_time || 'Exchange'})` : `Cached / Archived Quote (${selectedAsset.quote_time || 'time unavailable'})`}
                              </span>
                              <span style={{ opacity: 0.4 }}>•</span>
                              <span>
                                {selectedAsset.candle_is_live ? `Live 5m Candles (${selectedAsset.last_candle_time || ''})` : `${selectedAsset.candle_source === 'offline_historical_replay' ? 'Historical 5m Replay' : 'Cached / Archived 5m Candles'} (${selectedAsset.last_candle_time || 'time unavailable'})`}
                              </span>
                            </>
                          )}
                          <span style={{ opacity: 0.4 }}>•</span>
                          <span>USD</span>
                          {selectedAsset.regime && (
                            <>
                              <span style={{ opacity: 0.4 }}>•</span>
                              <span style={{
                                background: 'rgba(255, 255, 255, 0.06)',
                                border: '1px solid rgba(255, 255, 255, 0.08)',
                                padding: '1px 7px',
                                borderRadius: '4px',
                                color: '#E4E4E7',
                                fontSize: '0.75rem',
                                fontWeight: 500
                              }}>
                                Market Phase: {getPlainRegime(selectedAsset.regime)}
                              </span>
                            </>
                          )}
                        </div>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem' }}>
                        <button
                          onClick={() => {
                            setIsCopilotCollapsed(false);
                            sendChat(`Analyze ${selectedAsset.label} using the loaded price $${selectedAsset.price.toLocaleString()}, saved ${getPlainSignal(selectedAsset.signal)}, and loaded market phase ${getPlainRegime(selectedAsset.regime)}. Include the quote and model timestamps.`, selectedAsset, `✨ Analyze ${selectedAsset.label}`);
                          }}
                          style={{
                            background: 'rgba(255, 255, 255, 0.06)',
                            border: '1px solid var(--glass-border)',
                            color: '#FFFFFF',
                            borderRadius: '8px',
                            padding: '0.4rem 0.9rem',
                            fontSize: '0.82rem',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '0.4rem'
                          }}
                        >
                          <Bot size={15} /> Ask Advisor
                        </button>
                        <span className={`signal-badge signal-${selectedAsset.signal.toLowerCase()}`} style={{ fontSize: '0.88rem', padding: '0.4rem 1rem' }}>
                          <SignalIcon signal={selectedAsset.signal} />&nbsp;{getPlainSignal(selectedAsset.signal)}
                        </span>
                      </div>
                    </div>

                    <RecommendationCard asset={selectedAsset} onExplain={() => {
                      setIsCopilotCollapsed(false);
                      sendChat(`Why is the policy targeting ${selectedAsset.signal} for ${selectedAsset.label}? Explain using available evidence.`, selectedAsset, 'Why this signal?');
                    }} />

                    <div style={{ marginTop: '1.25rem' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem', flexWrap: 'wrap', gap: '0.5rem' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                          {!selectedAsset.quote_is_live || !selectedAsset.candle_is_live ? (
                            <span style={{ padding: '2px 7px', borderRadius: '4px', background: 'rgba(251, 191, 36, 0.12)', color: '#FBBF24', border: '1px solid rgba(251, 191, 36, 0.25)', fontWeight: 600 }}>
                              {selectedAsset.is_synthetic_candles ? 'Reconstructed Replay (Offline)' : `${selectedAsset.quote_is_live ? 'Live quote' : 'Cached / archived quote'} · ${selectedAsset.candle_is_live ? 'live 5m candles' : 'offline / archived 5m candles'}`}
                            </span>
                          ) : (
                            <span style={{ padding: '2px 7px', borderRadius: '4px', background: 'rgba(52, 211, 153, 0.12)', color: '#34D399', border: '1px solid rgba(52, 211, 153, 0.25)', fontWeight: 600 }}>
                              Live Quote and 5m Candles
                            </span>
                          )}
                          {selectedAsset.quote_time && (
                            <span>Quote: <strong style={{ color: '#E4E4E7' }}>{selectedAsset.quote_time}</strong></span>
                          )}
                          {selectedAsset.model_eval_time && (
                            <span>• Decision Model State: <strong style={{ color: '#E4E4E7' }}>{selectedAsset.model_eval_time}</strong></span>
                          )}
                        </div>
                        <div style={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          background: 'rgba(255, 255, 255, 0.04)',
                          borderRadius: '6px',
                          padding: '2px',
                          border: '1px solid rgba(255, 255, 255, 0.08)'
                        }}>
                          <button
                            onClick={() => setOverviewChartMode('line')}
                            style={{
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: '0.35rem',
                              background: overviewChartMode === 'line' ? '#FFFFFF' : 'transparent',
                              color: overviewChartMode === 'line' ? '#09090B' : 'var(--text-muted)',
                              border: 'none',
                              borderRadius: '4px',
                              padding: '0.22rem 0.6rem',
                              fontSize: '0.72rem',
                              fontWeight: 600,
                              cursor: 'pointer',
                              transition: 'all 0.15s ease'
                            }}
                          >
                            <Activity size={12} />
                            Line
                          </button>
                          <button
                            onClick={() => setOverviewChartMode('candles')}
                            style={{
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: '0.35rem',
                              background: overviewChartMode === 'candles' ? '#FFFFFF' : 'transparent',
                              color: overviewChartMode === 'candles' ? '#09090B' : 'var(--text-muted)',
                              border: 'none',
                              borderRadius: '4px',
                              padding: '0.22rem 0.6rem',
                              fontSize: '0.72rem',
                              fontWeight: 600,
                              cursor: 'pointer',
                              transition: 'all 0.15s ease'
                            }}
                          >
                            <BarChart2 size={12} />
                            Candles
                          </button>
                        </div>
                      </div>
                      {overviewChartMode === 'line' ? (
                        <AreaLineChart
                          candles={selectedAsset.candles}
                          data={selectedAsset.history}
                          color={getAssetDayTrend(selectedAsset).color}
                          height={380}
                          interactive={true}
                          isLive={selectedAsset.candle_is_live}
                          candleSource={selectedAsset.candle_source}
                        />
                      ) : (
                        <CandlestickChart
                          data={selectedAsset.candles || []}
                          width="100%"
                          height={380}
                          interactive={true}
                          isLive={selectedAsset.candle_is_live}
                          candleSource={selectedAsset.candle_source}
                        />
                      )}
                    </div>

                    {/* STRATEGY DEEP DIVE SECTION */}
                    <div style={{ marginTop: '1.75rem', padding: '1.5rem', borderLeft: '3px solid #34D399', background: 'rgba(255, 255, 255, 0.02)', borderRadius: '0 10px 10px 0' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '1rem' }}>
                        <div style={{ flex: 1 }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                            <Cpu size={18} color="#34D399" />
                            <h3 style={{ margin: 0, fontSize: '1.05rem', color: '#FFFFFF' }}>Strategy Lab</h3>
                          </div>
                          <p style={{ margin: '0 0 0.75rem 0', color: 'var(--text-secondary)', fontSize: '0.85rem', lineHeight: 1.5 }}>
                            Deep quantitative analysis for <strong style={{ color: '#E4E4E7' }}>{selectedAsset.label}</strong> — backtests, decision drivers, regime detection, walk-forward validation, and Monte Carlo stress tests.
                          </p>
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
                            {['📈 Backtest', '🔍 Key Drivers', '🌀 Regimes', '📊 Walk-Forward', '🎲 Monte Carlo'].map(tag => (
                              <span key={tag} style={{ fontSize: '0.74rem', padding: '0.2rem 0.55rem', borderRadius: '4px', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.08)', color: 'var(--text-secondary)', fontWeight: 500 }}>{tag}</span>
                            ))}
                          </div>
                        </div>
                        <button
                          onClick={() => handleOpenDeepAnalytics()}
                          style={{
                            background: '#FFFFFF',
                            color: '#000000',
                            border: 'none',
                            borderRadius: '8px',
                            padding: '0.6rem 1.4rem',
                            fontSize: '0.85rem',
                            fontWeight: 700,
                            cursor: 'pointer',
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '0.4rem',
                            boxShadow: '0 2px 10px rgba(255, 255, 255, 0.15)',
                            transition: 'all 0.18s ease',
                            whiteSpace: 'nowrap' as const,
                            flexShrink: 0,
                            marginTop: '0.25rem'
                          }}
                        >
                          Explore →
                        </button>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="glass-panel" style={{ padding: '3.5rem 2rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
                    <BarChart2 size={48} style={{ opacity: 0.25, marginBottom: '1rem' }} />
                    <h3 style={{ margin: '0 0 0.5rem 0', color: '#FFFFFF' }}>No Asset Selected</h3>
                    <p style={{ fontSize: '0.9rem', maxWidth: '420px', margin: '0 auto 1.5rem auto' }}>
                      Select an asset from the switcher above or return to the Trading Dashboard to inspect its overview.
                    </p>
                    <button
                      className="chat-suggested-btn"
                      onClick={() => setCopilotCenterTab('dashboard')}
                      style={{ width: 'auto', margin: '0 auto', display: 'inline-flex', alignItems: 'center', gap: '0.5rem', padding: '0.5rem 1.25rem' }}
                    >
                      ← Return to Trading Dashboard
                    </button>
                  </div>
                )}
              </>

  );

  const renderSimpleStrategyContent = () => {
    if (!selectedAsset) return null;
    const isBullish = selectedAsset.signal === 'BUY';
    const isBearish = selectedAsset.signal === 'SELL';
    const returnVal = backtestMetrics?.roi != null ? backtestMetrics.roi : null;
    const maxDdVal = backtestMetrics?.max_dd != null ? Math.abs(backtestMetrics.max_dd) : null;

    // Generate downsampled strategy performance (% gain/loss) curve
    const datesList = backtestMetrics?.dates || [];
    const historyList = backtestMetrics?.history || [];
    const baseVal = historyList.length > 0 ? historyList[0] : 10000;
    const stride = Math.max(1, Math.floor(historyList.length / 150));
    const equityChartData = [];
    for (let i = 0; i < historyList.length; i += stride) {
      const val = historyList[i];
      const rawDate = datesList[i] || '';
      let shortDate = `Bar ${i}`;
      const fullDate = rawDate || `Historical Bar #${i}`;
      if (rawDate) {
        if (rawDate.includes(' ')) {
          const parts = rawDate.split(' ');
          const datePart = parts[0];
          const timePart = parts[1].substring(0, 5);
          shortDate = `${datePart.slice(5)} ${timePart}`;
        } else {
          shortDate = rawDate.slice(5);
        }
      }
      const returnPct = baseVal > 0 ? ((val - baseVal) / baseVal) * 100 : 0;
      equityChartData.push({
        step: i,
        date: shortDate,
        fullDate: fullDate,
        returnPct: Number(returnPct.toFixed(2))
      });
    }

    if (historyList.length > 0 && (historyList.length - 1) % stride !== 0) {
      const lastIdx = historyList.length - 1;
      const val = historyList[lastIdx];
      const rawDate = datesList[lastIdx] || '';
      let shortDate = `Bar ${lastIdx}`;
      const fullDate = rawDate || `Historical Bar #${lastIdx}`;
      if (rawDate) {
        if (rawDate.includes(' ')) {
          const parts = rawDate.split(' ');
          const datePart = parts[0];
          const timePart = parts[1].substring(0, 5);
          shortDate = `${datePart.slice(5)} ${timePart}`;
        } else {
          shortDate = rawDate.slice(5);
        }
      }
      const returnPct = baseVal > 0 ? ((val - baseVal) / baseVal) * 100 : 0;
      equityChartData.push({
        step: lastIdx,
        date: shortDate,
        fullDate: fullDate,
        returnPct: Number(returnPct.toFixed(2))
      });
    }

    // Do not fabricate synthetic equity points when data is absent or calculating

    return (
      <div style={{ marginTop: '1.25rem', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
        {/* 1. HERO STRATEGY VERDICT */}
        <div style={{
          background: 'linear-gradient(135deg, rgba(52, 211, 153, 0.08) 0%, rgba(14, 165, 233, 0.04) 100%)',
          border: '1px solid rgba(52, 211, 153, 0.25)',
          borderRadius: '12px',
          padding: '1.5rem 1.75rem'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.75rem', marginBottom: '0.85rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
              <span style={{
                background: isBullish ? 'rgba(52, 211, 153, 0.2)' : isBearish ? 'rgba(248, 113, 113, 0.2)' : 'rgba(161, 161, 170, 0.2)',
                color: isBullish ? '#34D399' : isBearish ? '#F87171' : '#A1A1AA',
                border: `1px solid ${isBullish ? '#34D399' : isBearish ? '#F87171' : '#A1A1AA'}`,
                padding: '0.25rem 0.75rem',
                borderRadius: '20px',
                fontSize: '0.82rem',
                fontWeight: 700,
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.4rem'
              }}>
                <SignalIcon signal={selectedAsset.signal} /> Model Snapshot Target: {getPlainSignal(selectedAsset.signal)}
              </span>
              <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                Chosen-action probability: <strong style={{ color: '#FFFFFF' }}>{(selectedAsset.confidence > 1 ? selectedAsset.confidence : selectedAsset.confidence * 100).toFixed(0)}%</strong>
              </span>
            </div>
            <div style={{ fontSize: '0.78rem', color: '#34D399', background: 'rgba(52, 211, 153, 0.1)', padding: '0.2rem 0.6rem', borderRadius: '6px', fontWeight: 600 }}>
              Saved policy snapshot (historical input)
            </div>
          </div>

          <h3 style={{ margin: '0 0 0.5rem 0', fontSize: '1.25rem', color: '#FFFFFF', lineHeight: 1.3 }}>
            {isBullish
              ? `The policy targets Long for ${selectedAsset.label}.`
              : isBearish
              ? `The policy targets Short for ${selectedAsset.label}.`
              : `The policy targets Cash for ${selectedAsset.label}.`}
          </h3>
          <p style={{ margin: 0, fontSize: '0.9rem', color: 'var(--text-secondary)', lineHeight: 1.5, maxWidth: '850px' }}>
            {loadingBacktest ? (
              'Calculating historical backtest performance...'
            ) : returnVal === null ? (
              'Historical backtest data is currently unavailable for this asset.'
            ) : returnVal < 0 ? (
              (() => {
                const bhVal = backtestMetrics?.bh_return != null ? backtestMetrics.bh_return : null;
                const bhFormatted = bhVal != null ? `${bhVal >= 0 ? '+' : ''}${Number(bhVal).toFixed(2)}%` : 'N/A';
                return `During this backtest period, underlying asset return was ${bhFormatted}; strategy recorded ${Number(returnVal).toFixed(2)}% net return with a maximum drawdown of -${maxDdVal != null ? Number(maxDdVal).toFixed(2) : '0.00'}% under execution and fee friction.`;
              })()
            ) : (
              `Trained on multi-year exchange data and backtested across historical market cycles. In this historical simulation, the strategy returned +${Number(returnVal).toFixed(2)}% with a maximum drawdown of -${maxDdVal != null ? Number(maxDdVal).toFixed(2) : '0.00'}%.`
            )}
          </p>
        </div>

        {/* 2. THE THREE CORE DECISION CARDS */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.25rem' }}>
          
          {/* Card A: saved model snapshot */}
          {(() => {
            const card1 = getCard1Details(selectedAsset.regime, selectedAsset.signal);
            return (
              <div className="glass-panel" style={{ padding: '1.4rem', borderRadius: '10px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
                    <Activity size={18} color="#60A5FA" />
                    <h4 style={{ margin: 0, fontSize: '1rem', color: '#FFFFFF' }}>1. What Did the Model Target?</h4>
                  </div>
                  <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: '0.75rem', borderRadius: '8px', marginBottom: '0.75rem', border: '1px solid rgba(255, 255, 255, 0.06)' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>LOADED MARKET REGIME</div>
                    <div style={{ fontSize: '1.05rem', fontWeight: 700, color: '#60A5FA', marginTop: '0.2rem' }}>
                      {card1.regimeDisplay}
                    </div>
                  </div>
                  <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.5, margin: 0 }}>
                    {card1.explanation}
                  </p>
                </div>
                <div style={{ marginTop: '1rem', paddingTop: '0.75rem', borderTop: '1px solid rgba(255, 255, 255, 0.08)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Target Allocation:</span>
                  <span style={{ fontSize: '0.85rem', fontWeight: 700, color: '#34D399' }}>
                    {card1.allocation}
                  </span>
                </div>
              </div>
            );
          })()}

          {/* Card B: Why did the AI make this call? */}
          <div className="glass-panel" style={{ padding: '1.4rem', borderRadius: '10px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Search size={18} color="#34D399" />
                  <h4 style={{ margin: 0, fontSize: '1rem', color: '#FFFFFF' }}>2. Why Did the AI Make This Call?</h4>
                </div>
                <span style={{ fontSize: '0.72rem', color: '#34D399', background: 'rgba(52, 211, 153, 0.1)', padding: '0.15rem 0.45rem', borderRadius: '4px', fontWeight: 600 }}>
                  Key Decision Drivers
                </span>
              </div>
              <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginBottom: '0.85rem' }}>
                Evaluated from multi-indicator decision weights:
              </div>

              {loadingShap ? (
                <div style={{ padding: '1.5rem 0', textAlign: 'center', color: 'var(--text-secondary)' }}>
                  <div className="typing-indicator" style={{ margin: 'auto' }}><span></span><span></span><span></span></div>
                  <div style={{ fontSize: '0.78rem', marginTop: '0.6rem' }}>Calculating key decision drivers for {selectedAsset.label}...</div>
                </div>
              ) : shapData && shapData.shap && shapData.shap.length > 0 ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
                  {(() => {
                    const totalAbs = shapData.shap.reduce((acc: number, x: ShapItem) => acc + Math.abs(x.value), 0) || 1e-6;
                    const validItems = shapData.shap
                      .map((item: ShapItem) => {
                        const pct = item.importance_pct != null
                          ? item.importance_pct
                          : Math.round((Math.abs(item.value) / totalAbs) * 1000) / 10;
                        return { ...item, pct };
                      })
                      .filter(item => Math.abs(item.pct) >= 0.5);
                    const topItems = validItems.length > 0 ? validItems.slice(0, 3) : shapData.shap.slice(0, 3).map(x => ({ ...x, pct: 0 }));

                    return topItems.map((item, idx: number) => {
                      const cfg = cleanFeatureLabel(item.feature);
                      const pct = item.pct;
                      const isPositive = item.value >= 0;
                      const barColor = isPositive ? '#34D399' : '#F87171';

                      return (
                        <div key={idx}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', marginBottom: '0.2rem' }}>
                            <span style={{ color: '#E4E4E7', fontWeight: 600 }}>{cfg.icon} {cfg.label}</span>
                            <span style={{ color: barColor, fontWeight: 700 }}>
                              {isPositive ? '+' : '-'}{pct.toFixed(1)}% Impact
                            </span>
                          </div>
                          <div style={{ width: '100%', height: '6px', background: 'rgba(255,255,255,0.08)', borderRadius: '3px', overflow: 'hidden' }}>
                            <div style={{ width: `${Math.min(100, Math.max(6, pct))}%`, height: '100%', background: barColor, borderRadius: '3px', transition: 'width 0.4s ease' }}></div>
                          </div>
                        </div>
                      );
                    });
                  })()}
                </div>
              ) : (
                <div style={{ padding: '1rem', textAlign: 'center', background: 'rgba(255,255,255,0.02)', borderRadius: '6px', border: '1px dashed var(--glass-border)' }}>
                  <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginBottom: '0.6rem' }}>Decision drivers data pending</div>
                  <button
                    onClick={() => fetchAnalytics(selectedAsset.label, selectedRegime, true)}
                    style={{
                      background: 'rgba(255, 255, 255, 0.08)',
                      color: '#FFFFFF',
                      border: '1px solid var(--glass-border)',
                      borderRadius: '6px',
                      padding: '0.35rem 0.8rem',
                      fontSize: '0.76rem',
                      cursor: 'pointer'
                    }}
                  >
                    Compute Key Drivers
                  </button>
                </div>
              )}
            </div>

            {(() => {
              const topItem = shapData?.shap?.[0];
              const totalAbs = shapData?.shap?.reduce((acc: number, x: ShapItem) => acc + Math.abs(x.value), 0) || 1e-6;
              const topCfg = topItem ? cleanFeatureLabel(topItem.feature) : null;
              const topPct = topItem ? (topItem.importance_pct != null ? topItem.importance_pct : Math.round((Math.abs(topItem.value) / totalAbs) * 1000) / 10).toFixed(1) : '0';
              const isTopPos = topItem ? topItem.value >= 0 : true;

              let takeawayText: string;
              if (topCfg) {
                takeawayText = `In this saved model snapshot, ${topCfg.label} had the largest displayed attribution (${isTopPos ? '+' : '-'}${topPct}% relative magnitude). This does not establish a current market trend.`;
              } else {
                takeawayText = 'Computed feature drivers are not available for this model snapshot.';
              }

              return (
                <div style={{ marginTop: '0.85rem', padding: '0.45rem 0.65rem', background: isBullish ? 'rgba(52, 211, 153, 0.08)' : isBearish ? 'rgba(248, 113, 113, 0.08)' : 'rgba(251, 191, 36, 0.08)', borderRadius: '6px', fontSize: '0.76rem', color: '#E4E4E7' }}>
                  💡 <strong>Model Snapshot Takeaway:</strong> {takeawayText}
                </div>
              );
            })()}
          </div>

          {/* Card C: Interactive Stress Simulator */}
          <div className="glass-panel" style={{ padding: '1.4rem', borderRadius: '10px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
                <Shield size={18} color="#FBBF24" />
                <h4 style={{ margin: 0, fontSize: '1rem', color: '#FFFFFF' }}>3. Historical Walk-Forward Stress Tests & Simulation</h4>
              </div>
              <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
                Historical Walk-Forward Folds (BTC/USDT 1H Saved Experiment):
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.35rem', marginBottom: '0.85rem' }}>
                {[
                  { id: 'normal', label: 'Fold 1 (Rows 9k–19k)' },
                  { id: 'choppy', label: 'Fold 3 (Rows 28k–38k)' },
                  { id: 'crash', label: 'Fold 2 (Rows 19k–28k)' }
                ].map(s => (
                  <button
                    key={s.id}
                    onClick={() => setSimpleScenario(s.id as 'normal' | 'choppy' | 'crash')}
                    style={{
                      background: simpleScenario === s.id ? '#FFFFFF' : 'rgba(255, 255, 255, 0.05)',
                      color: simpleScenario === s.id ? '#000000' : 'var(--text-secondary)',
                      border: simpleScenario === s.id ? '1px solid #FFFFFF' : '1px solid rgba(255, 255, 255, 0.1)',
                      borderRadius: '6px',
                      padding: '0.4rem 0.2rem',
                      fontSize: '0.78rem',
                      fontWeight: simpleScenario === s.id ? 700 : 500,
                      cursor: 'pointer',
                      textAlign: 'center',
                      transition: 'all 0.15s ease'
                    }}
                  >
                    {s.label}
                  </button>
                ))}
              </div>

              <div style={{
                background: 'rgba(0, 0, 0, 0.3)',
                borderRadius: '8px',
                padding: '0.75rem',
                border: '1px solid rgba(255, 255, 255, 0.06)'
              }}>
                {simpleScenario === 'normal' && (
                  <div>
                    <div style={{ fontSize: '0.82rem', fontWeight: 700, color: '#34D399', marginBottom: '0.25rem' }}>
                      Fold 1: Low-Volatility Expansion (Rows 9,626–19,252)
                    </div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
                      Out-of-sample test window (Rows 9,626–19,252; reconstructed dates Feb 2021 – Mar 2022): strategy recorded <strong>+6.31% ROI</strong> vs <strong>+0.52% benchmark</strong> with <strong>-6.74% max drawdown</strong> across 2 trades.
                    </div>
                  </div>
                )}
                {simpleScenario === 'choppy' && (
                  <div>
                    <div style={{ fontSize: '0.82rem', fontWeight: 700, color: '#FBBF24', marginBottom: '0.25rem' }}>
                      Fold 3: Sideways Transition (Rows 28,878–38,504)
                    </div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
                      Out-of-sample test window (Rows 28,878–38,504; reconstructed dates May 2023 – Jun 2024): the policy targeted 100% Cash (0.00% ROI, 0 trades), avoiding transaction friction while the underlying market returned +123.79%.
                    </div>
                  </div>
                )}
                {simpleScenario === 'crash' && (
                  <div>
                    <div style={{ fontSize: '0.82rem', fontWeight: 700, color: '#F87171', marginBottom: '0.25rem' }}>
                      Fold 2: High-Volatility Downturn (Rows 19,252–28,878)
                    </div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
                      Out-of-sample test window (Rows 19,252–28,878; reconstructed dates Mar 2022 – May 2023, Terra/Luna &amp; FTX bear market): strategy recorded <strong>+44.51% ROI</strong> vs <strong>-20.65% benchmark</strong> with <strong>-14.35% max drawdown</strong> across 65 trades.
                    </div>
                  </div>
                )}
              </div>
            </div>

            <div style={{ marginTop: '0.85rem', paddingTop: '0.65rem', borderTop: '1px solid rgba(255, 255, 255, 0.06)', display: 'flex', flexDirection: 'column', gap: '0.35rem', fontSize: '0.78rem' }}>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: '0.2rem', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                Forward-Looking Monte Carlo Projection
              </div>
              {!mcData && !loadingMC && <p style={{ margin: '0.3rem 0', color: 'var(--text-secondary)' }}>Optional simulation — runs only when requested.</p>}
              <button disabled={loadingMC} onClick={() => fetchMonteCarlo(selectedAsset.label)} className="detail-tab">
                {loadingMC ? 'Simulating...' : 'Run Simulation'}
              </button>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ color: 'var(--text-muted)' }}>Simulation Success Rate:</span>
                <span style={{ color: loadingMC ? 'var(--text-muted)' : (mcData?.summary?.success_rate != null ? '#34D399' : 'var(--text-muted)'), fontWeight: 700 }}>
                  {loadingMC ? (
                    'Calculating...'
                  ) : mcData?.summary?.success_rate != null ? (
                    (() => {
                      const val = Number(mcData.summary.success_rate);
                      const safeSuccessRate = Math.min(100.0, Math.max(0.0, isNaN(val) ? 0 : val));
                      return `${safeSuccessRate.toFixed(1)}% Profitable Simulations`;
                    })()
                  ) : (
                    mcData ? 'Unavailable' : 'Run simulation to estimate'
                  )}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ color: 'var(--text-muted)' }}>5th Percentile Return:</span>
                <span style={{ color: loadingMC ? 'var(--text-muted)' : (mcData?.summary?.var_5th != null ? (mcData.summary.var_5th >= 0 ? '#34D399' : '#F87171') : 'var(--text-muted)'), fontWeight: 700 }}>
                  {loadingMC ? (
                    'Calculating...'
                  ) : mcData?.summary?.var_5th != null ? (
                    `${mcData.summary.var_5th.toFixed(1)}%`
                  ) : (
                    mcData ? 'Not available' : 'Run simulation to estimate'
                  )}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* 3. SIMULATED STRATEGY PERFORMANCE GRAPH (% GAIN / LOSS) */}
        <div style={{
          background: 'rgba(255, 255, 255, 0.02)',
          border: '1px solid var(--glass-border)',
          borderRadius: '12px',
          padding: '1.25rem'
        }}>
          {/* Header with Visual Status Badges and Metrics */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.85rem', flexWrap: 'wrap', gap: '0.75rem' }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flexWrap: 'wrap' }}>
                <TrendingUp size={18} color={returnVal != null && returnVal >= 0 ? '#34D399' : returnVal != null ? '#F87171' : 'var(--text-muted)'} />
                <h4 style={{ margin: 0, fontSize: '1rem', color: '#FFFFFF', fontWeight: 600 }}>
                  AI Model Historical Track Record
                </h4>
                <span style={{
                  fontSize: '0.72rem',
                  fontWeight: 600,
                  color: '#34D399',
                  background: 'rgba(52, 211, 153, 0.12)',
                  border: '1px solid rgba(52, 211, 153, 0.25)',
                  padding: '0.15rem 0.55rem',
                  borderRadius: '12px'
                }}>
                  ● Historical Backtest
                </span>
              </div>
              <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)', marginTop: '0.35rem' }}>
                Cumulative % gain/loss across historical exchange candles under 0.1% fee & 0.05% slippage friction
              </div>
            </div>

            {/* Quick Metrics */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flexWrap: 'wrap' }}>
              <div style={{
                background: returnVal != null && returnVal >= 0 ? 'rgba(52, 211, 153, 0.08)' : 'rgba(248, 113, 113, 0.08)',
                border: `1px solid ${returnVal != null && returnVal >= 0 ? 'rgba(52, 211, 153, 0.25)' : 'rgba(248, 113, 113, 0.25)'}`,
                padding: '0.35rem 0.75rem',
                borderRadius: '8px',
                textAlign: 'right'
              }}>
                <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Net Return</div>
                <div style={{ fontSize: '1.05rem', fontWeight: 700, color: returnVal != null && returnVal >= 0 ? '#34D399' : returnVal != null ? '#F87171' : 'var(--text-muted)' }}>
                  {loadingBacktest ? 'Calculating...' : (returnVal != null ? `${returnVal >= 0 ? '+' : ''}${Number(returnVal).toFixed(2)}%` : 'Unavailable')}
                </div>
              </div>

              <div style={{
                background: 'rgba(248, 113, 113, 0.08)',
                border: '1px solid rgba(248, 113, 113, 0.25)',
                padding: '0.35rem 0.75rem',
                borderRadius: '8px',
                textAlign: 'right'
              }}>
                <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Max Drawdown</div>
                <div style={{ fontSize: '1.05rem', fontWeight: 700, color: maxDdVal != null ? '#F87171' : 'var(--text-muted)' }}>
                  {loadingBacktest ? 'Calculating...' : (maxDdVal != null ? `-${Number(maxDdVal).toFixed(2)}%` : 'Unavailable')}
                </div>
              </div>
            </div>
          </div>

          <div style={{ height: '250px', width: '100%' }}>
            {loadingBacktest ? (
              <div style={{ display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                <div className="typing-indicator" style={{ marginRight: '0.75rem' }}><span></span><span></span><span></span></div>
                Calculating historical backtest equity curve...
              </div>
            ) : equityChartData.length === 0 ? (
              <div style={{ display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                Historical equity data unavailable
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={equityChartData}>
                  <defs>
                    <linearGradient id="strategyReturnGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={returnVal != null && returnVal >= 0 ? '#34D399' : '#F87171'} stopOpacity={0.35} />
                      <stop offset="95%" stopColor={returnVal != null && returnVal >= 0 ? '#34D399' : '#F87171'} stopOpacity={0.0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="date" stroke="var(--text-muted)" fontSize={10} minTickGap={35} />
                  <YAxis
                    stroke="var(--text-muted)"
                    fontSize={10}
                    domain={['auto', 'auto']}
                    tickFormatter={(v) => `${Number(v) >= 0 ? '+' : ''}${Number(v).toFixed(1)}%`}
                  />
                  <RechartsTooltip
                    contentStyle={{ background: '#18181B', borderColor: 'rgba(255, 255, 255, 0.15)', borderRadius: '8px', color: '#FFFFFF' }}
                    formatter={(v: unknown) => [`${Number(v) >= 0 ? '+' : ''}${Number(v).toFixed(2)}%`, 'Strategy Return']}
                    labelFormatter={(_, payload) => {
                      const item = payload?.[0]?.payload;
                      return item?.fullDate ? `Date: ${item.fullDate}` : `Bar #${item?.step}`;
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="returnPct"
                    stroke={returnVal != null && returnVal >= 0 ? '#34D399' : '#F87171'}
                    strokeWidth={2}
                    fill="url(#strategyReturnGrad)"
                    name="Strategy Return"
                    dot={false}
                    isAnimationActive={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </div>

          <div style={{ marginTop: '0.65rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.72rem', color: 'var(--text-muted)', flexWrap: 'wrap', gap: '0.5rem' }}>
            <span>* Simulated algorithmic track record across past market candles.</span>
            <span>Fee model: 0.10% commission • 0.05% slippage</span>
          </div>
        </div>

        {/* 4. FOOTER CALLOUT TO QUANT / ADVANCED */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0.9rem 1.25rem',
          background: 'rgba(96, 165, 250, 0.05)',
          border: '1px solid rgba(96, 165, 250, 0.18)',
          borderRadius: '10px',
          flexWrap: 'wrap',
          gap: '0.75rem'
        }}>
          <div style={{ fontSize: '0.84rem', color: '#E4E4E7' }}>
            🔬 <strong>Institutional Quant Inspection:</strong> Switch to Advanced Mode for 5-Fold Walk-Forward matrix, anti-overfitting evaluations, and Monte Carlo risk cones.
          </div>
          <button
            onClick={() => handleToggleViewMode('advanced')}
            style={{
              background: '#60A5FA',
              color: '#09090B',
              border: 'none',
              borderRadius: '6px',
              padding: '0.45rem 1rem',
              fontSize: '0.82rem',
              fontWeight: 700,
              cursor: 'pointer',
              whiteSpace: 'nowrap'
            }}
          >
            Switch to 🔬 Advanced Mode →
          </button>
        </div>
      </div>
    );
  };

  const renderDeepAnalyticsContent = () => (
              <>
                {/* Top Navigation Bar: Back Button */}
                <div style={{ display: 'flex', alignItems: 'center', marginBottom: '1.25rem' }}>
                  <button
                    onClick={() => setCopilotCenterTab('overview')}
                    style={{
                      background: 'rgba(255, 255, 255, 0.05)',
                      border: '1px solid var(--glass-border)',
                      color: '#FFFFFF',
                      borderRadius: '8px',
                      padding: '0.45rem 1rem',
                      fontSize: '0.85rem',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.5rem',
                      transition: 'all 0.2s ease'
                    }}
                  >
                    <ArrowLeft size={16} /> Back to Asset Overview
                  </button>
                </div>

                {selectedAsset ? (
                  <div className="glass-panel" style={{ padding: '2rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
                      <div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                          <Cpu size={22} color="#34D399" />
                          <h2 style={{ margin: 0, fontSize: '1.6rem' }}>Strategy Lab: {selectedAsset.label}</h2>
                        </div>
                        <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                          Simulated Performance, Key Decision Drivers, Market Phases & Stress Testing
                        </div>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', flexWrap: 'wrap' }}>
                        {/* View Mode Segmented Control */}
                        <div style={{
                          display: 'inline-flex',
                          background: 'rgba(255, 255, 255, 0.04)',
                          border: '1px solid rgba(255, 255, 255, 0.08)',
                          borderRadius: '8px',
                          padding: '3px',
                          gap: '2px'
                        }}>
                          <button
                            onClick={() => handleToggleViewMode('simple')}
                            style={{
                              background: viewMode === 'simple' ? 'rgba(255, 255, 255, 0.12)' : 'transparent',
                              color: viewMode === 'simple' ? '#FFFFFF' : 'var(--text-secondary)',
                              border: viewMode === 'simple' ? '1px solid rgba(255, 255, 255, 0.1)' : '1px solid transparent',
                              borderRadius: '6px',
                              padding: '0.35rem 0.75rem',
                              fontSize: '0.8rem',
                              fontWeight: viewMode === 'simple' ? 600 : 400,
                              cursor: 'pointer',
                              display: 'flex',
                              alignItems: 'center',
                              gap: '0.35rem',
                              transition: 'all 0.15s ease'
                            }}
                          >
                            <Sparkles size={13} style={{ color: viewMode === 'simple' ? '#34D399' : 'inherit' }} />
                            Simple Mode
                          </button>
                          <button
                            onClick={() => handleToggleViewMode('advanced')}
                            style={{
                              background: viewMode === 'advanced' ? 'rgba(255, 255, 255, 0.12)' : 'transparent',
                              color: viewMode === 'advanced' ? '#FFFFFF' : 'var(--text-secondary)',
                              border: viewMode === 'advanced' ? '1px solid rgba(255, 255, 255, 0.1)' : '1px solid transparent',
                              borderRadius: '6px',
                              padding: '0.35rem 0.75rem',
                              fontSize: '0.8rem',
                              fontWeight: viewMode === 'advanced' ? 600 : 400,
                              cursor: 'pointer',
                              display: 'flex',
                              alignItems: 'center',
                              gap: '0.35rem',
                              transition: 'all 0.15s ease'
                            }}
                          >
                            <Activity size={13} style={{ color: viewMode === 'advanced' ? '#60A5FA' : 'inherit' }} />
                            Advanced (Quant)
                          </button>
                        </div>

                        <button
                          onClick={() => {
                            if (selectedAsset) {
                              fetchAnalytics(selectedAsset.label, selectedRegime, true);
                              fetchWalkForward();
                            }
                          }}
                          disabled={loadingAnalytics || loadingWFV || loadingMC}
                          style={{
                            background: 'rgba(255, 255, 255, 0.04)',
                            border: '1px solid rgba(255, 255, 255, 0.08)',
                            color: '#E4E4E7',
                            borderRadius: '8px',
                            padding: '0.4rem 0.85rem',
                            fontSize: '0.8rem',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '0.4rem',
                            transition: 'all 0.15s ease'
                          }}
                        >
                          <RotateCcw size={13} /> Refresh
                        </button>
                        <button
                          onClick={() => {
                            setIsCopilotCollapsed(false);
                            sendChat(`Analyze ${selectedAsset.label}: summarize policy signal, top feature driver, and risk outlook in 2-3 sentences.`, selectedAsset);
                          }}
                          style={{
                            background: 'rgba(255, 255, 255, 0.08)',
                            border: '1px solid rgba(255, 255, 255, 0.14)',
                            color: '#FFFFFF',
                            borderRadius: '8px',
                            padding: '0.4rem 0.85rem',
                            fontSize: '0.8rem',
                            cursor: 'pointer',
                            fontWeight: 600,
                            display: 'flex',
                            alignItems: 'center',
                            gap: '0.4rem',
                            transition: 'all 0.15s ease'
                          }}
                        >
                          <Bot size={14} color="#60A5FA" /> Ask Advisor
                        </button>
                      </div>
                    </div>

                    {/* Executive Performance Ribbon */}
                    {viewMode === 'simple' ? (
                      <>
                        <div className="strategy-kpi-grid-5" style={{
                          marginBottom: backtestMetrics?.roi != null && backtestMetrics.roi < 0 ? '0.6rem' : '1rem'
                        }}>
                          {(() => {
                            const sharpeVal = backtestMetrics?.sharpe;
                            const gradeColor = sharpeVal == null ? 'var(--text-muted)' : sharpeVal >= 1 ? '#34D399' : sharpeVal >= 0 ? '#FBBF24' : '#F87171';
                            const gradeLabel = sharpeVal == null ? '—' : sharpeVal >= 2 ? '🌟 Exceptional' : sharpeVal >= 1 ? '✅ Strong' : sharpeVal >= 0 ? '⚠️ Mixed' : '⚠️ Weak';
                            const gradeDesc = sharpeVal == null ? 'Calculating historical Sharpe ratio' : `Historical Sharpe ratio: ${Number(sharpeVal).toFixed(2)}`;

                            return (
                              <div className="glass-panel" style={{ padding: '0.75rem 1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '8px', border: '1px solid var(--glass-border)' }}>
                                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Risk-Adjusted Result</div>
                                <div style={{ fontSize: '1.15rem', fontWeight: 700, color: gradeColor, marginTop: '0.2rem' }}>
                                  {gradeLabel}
                                </div>
                                <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '0.15rem' }}>
                                  {gradeDesc}
                                </div>
                              </div>
                            );
                          })()}

                          <div className="glass-panel" style={{ padding: '0.75rem 1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '8px', border: '1px solid var(--glass-border)' }}>
                            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Worst Temporary Dip</div>
                            <div style={{ fontSize: '1.2rem', fontWeight: 700, color: '#F87171', marginTop: '0.2rem' }}>
                              {backtestMetrics?.max_dd != null ? `${Number(backtestMetrics.max_dd).toFixed(2)}%` : '—'}
                            </div>
                            <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '0.15rem' }}>Max peak-to-trough drop</div>
                          </div>

                          <div className="glass-panel" style={{ padding: '0.75rem 1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '8px', border: '1px solid var(--glass-border)' }}>
                            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Simulated Return</div>
                            <div style={{ fontSize: '1.2rem', fontWeight: 700, color: (backtestMetrics?.roi ?? 0) >= 0 ? '#34D399' : '#F87171', marginTop: '0.2rem' }}>
                              {backtestMetrics?.roi != null ? `${backtestMetrics.roi >= 0 ? '+' : ''}${Number(backtestMetrics.roi).toFixed(2)}%` : '—'}
                            </div>
                            <div style={{ fontSize: '0.72rem', color: backtestMetrics?.roi != null && backtestMetrics.roi < 0 ? '#93C5FD' : 'var(--text-secondary)', marginTop: '0.15rem' }}>
                              {backtestMetrics?.bh_return != null
                                ? `Asset: ${Number(backtestMetrics.bh_return).toFixed(1)}% · ${backtestMetrics.trades ?? 0} trades`
                                : `Across ${backtestMetrics?.trades ?? 0} trades`}
                            </div>
                          </div>

                          <div className="glass-panel" style={{ padding: '0.75rem 1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '8px', border: '1px solid var(--glass-border)' }}>
                            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Walk-Forward Validation</div>
                            <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#A78BFA', marginTop: '0.2rem' }}>
                              5 Folds Tested
                            </div>
                            <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '0.15rem' }}>Out-of-sample historical blocks</div>
                          </div>

                          <div className="glass-panel" style={{ padding: '0.75rem 1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '8px', border: '1px solid var(--glass-border)' }}>
                            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Market Environment</div>
                            <div style={{ fontSize: '0.92rem', fontWeight: 700, color: '#E4E4E7', marginTop: '0.2rem', whiteSpace: 'normal', lineHeight: 1.25 }}>
                              {getPlainRegime(selectedAsset.regime)}
                            </div>
                            <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '0.15rem' }}>{selectedAsset.candle_is_live ? 'Detected from live 5m candles' : 'Detected from offline / archived data'}{selectedAsset.model_eval_time ? ` · evaluated ${selectedAsset.model_eval_time}` : ''}</div>
                          </div>
                        </div>

                        {/* Benchmark context for negative returns */}
                        {backtestMetrics?.roi != null && backtestMetrics.roi < 0 && backtestMetrics.bh_return != null && (
                          <div style={{
                            background: 'rgba(59, 130, 246, 0.08)',
                            border: '1px solid rgba(59, 130, 246, 0.25)',
                            borderRadius: '8px',
                            padding: '0.7rem 1rem',
                            marginBottom: '1rem',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '0.65rem',
                            fontSize: '0.84rem',
                            color: '#93C5FD'
                          }}>
                            <Shield size={16} color="#60A5FA" style={{ flexShrink: 0 }} />
                            <span>
                              <strong>Historical benchmark comparison:</strong>{' '}
                              Strategy returned {Number(backtestMetrics.roi).toFixed(2)}% versus {backtestMetrics.bh_return >= 0 ? '+' : ''}{Number(backtestMetrics.bh_return).toFixed(2)}% for the underlying asset.
                            </span>
                          </div>
                        )}
                      </>
                    ) : (
                      <div className="strategy-kpi-grid-6">
                        <div className="glass-panel" style={{ padding: '0.75rem 1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '8px', border: '1px solid var(--glass-border)' }}>
                          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Sharpe Ratio</div>
                          <div style={{ fontSize: '1.2rem', fontWeight: 700, color: (backtestMetrics?.sharpe ?? 0) >= 1 ? '#34D399' : (backtestMetrics?.sharpe ?? 0) >= 0 ? '#FBBF24' : '#F87171', marginTop: '0.2rem' }}>
                            {backtestMetrics?.sharpe != null ? Number(backtestMetrics.sharpe).toFixed(2) : '—'} <span style={{ fontSize: '0.7rem', fontWeight: 600, opacity: 0.85 }}>{(backtestMetrics?.sharpe ?? 0) >= 1 ? '(Strong)' : (backtestMetrics?.sharpe ?? 0) >= 0 ? '(Moderate)' : '(Sub-Optimal)'}</span>
                          </div>
                          <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', marginTop: '0.15rem' }}>Return vs volatility swings</div>
                        </div>

                        <div className="glass-panel" style={{ padding: '0.75rem 1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '8px', border: '1px solid var(--glass-border)' }}>
                          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Worst Historical Dip</div>
                          <div style={{ fontSize: '1.2rem', fontWeight: 700, color: '#F87171', marginTop: '0.2rem' }}>
                            {backtestMetrics?.max_dd != null ? `${Number(backtestMetrics.max_dd).toFixed(2)}%` : '—'}
                          </div>
                          <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', marginTop: '0.15rem' }}>Max peak-to-trough drop</div>
                        </div>

                        <div className="glass-panel" style={{ padding: '0.75rem 1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '8px', border: '1px solid var(--glass-border)' }}>
                          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Sortino Ratio</div>
                          <div style={{ fontSize: '1.2rem', fontWeight: 700, color: '#60A5FA', marginTop: '0.2rem' }}>
                            {backtestMetrics?.sortino != null ? Number(backtestMetrics.sortino).toFixed(2) : '—'}
                          </div>
                          <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', marginTop: '0.15rem' }}>Downside-only risk score</div>
                        </div>

                        <div className="glass-panel" style={{ padding: '0.75rem 1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '8px', border: '1px solid var(--glass-border)' }}>
                          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Total Backtest ROI</div>
                          <div style={{ fontSize: '1.2rem', fontWeight: 700, color: (backtestMetrics?.roi ?? 0) >= 0 ? '#34D399' : '#F87171', marginTop: '0.2rem' }}>
                            {backtestMetrics?.roi != null ? `${backtestMetrics.roi >= 0 ? '+' : ''}${Number(backtestMetrics.roi).toFixed(2)}%` : '—'}
                          </div>
                          <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', marginTop: '0.15rem' }}>Net of simulated friction</div>
                        </div>

                        <div className="glass-panel" style={{ padding: '0.75rem 1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '8px', border: '1px solid var(--glass-border)' }}>
                          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Multi-Cycle Sharpe</div>
                          <div style={{ fontSize: '1.2rem', fontWeight: 700, color: '#A78BFA', marginTop: '0.2rem' }}>
                            {wfvData && wfvData.length > 0 ? (wfvData.map(r => Number(r['Out-of-Sample Sharpe']) || 0).reduce((a, b) => a + b, 0) / wfvData.length).toFixed(2) : '—'}
                          </div>
                          <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', marginTop: '0.15rem' }}>Tested on 5 unseen eras</div>
                        </div>

                        <div className="glass-panel" style={{ padding: '0.75rem 1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '8px', border: '1px solid var(--glass-border)' }}>
                          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Market Climate</div>
                          <div style={{ fontSize: '0.92rem', fontWeight: 700, color: '#E4E4E7', marginTop: '0.2rem', whiteSpace: 'normal', lineHeight: 1.25 }}>
                            {getPlainRegime(selectedAsset.regime)}
                          </div>
                          <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', marginTop: '0.15rem' }}>HMM latent phase</div>
                        </div>
                      </div>
                    )}

                    {viewMode === 'simple' ? (
                      renderSimpleStrategyContent()
                    ) : (
                      <>
                        {/* Sub-Tab Filter Toolbar */}
                        <div className="detail-tabs-container" style={{ position: 'relative', zIndex: 10, padding: '0.4rem 0.5rem', borderRadius: '10px', marginTop: '1.25rem', marginBottom: '1.25rem', display: 'flex', gap: '0.4rem', border: '1px solid var(--glass-border)', background: 'rgba(16, 16, 20, 0.65)', backdropFilter: 'blur(12px)', overflowX: 'auto' }}>
                      <button onClick={() => setDeepSubTab('all')} className={`detail-tab ${deepSubTab === 'all' ? 'active' : ''}`} style={{ cursor: 'pointer', padding: '0.4rem 0.9rem' }}>🌐 All Modules</button>
                      <button onClick={() => setDeepSubTab('backtest')} className={`detail-tab ${deepSubTab === 'backtest' ? 'active' : ''}`} style={{ cursor: 'pointer', padding: '0.4rem 0.9rem' }}>📈 Backtest & Equity</button>
                      <button onClick={() => setDeepSubTab('shap')} className={`detail-tab ${deepSubTab === 'shap' ? 'active' : ''}`} style={{ cursor: 'pointer', padding: '0.4rem 0.9rem' }}>🔍 SHAP Explainability</button>
                      <button onClick={() => setDeepSubTab('regime')} className={`detail-tab ${deepSubTab === 'regime' ? 'active' : ''}`} style={{ cursor: 'pointer', padding: '0.4rem 0.9rem' }}>🌦️ Market Climate</button>
                      <button onClick={() => setDeepSubTab('wfv')} className={`detail-tab ${deepSubTab === 'wfv' ? 'active' : ''}`} style={{ cursor: 'pointer', padding: '0.4rem 0.9rem' }}>📊 Multi-Year Validation</button>
                      <button onClick={() => setDeepSubTab('montecarlo')} className={`detail-tab ${deepSubTab === 'montecarlo' ? 'active' : ''}`} style={{ cursor: 'pointer', padding: '0.4rem 0.9rem' }}>🎲 Monte Carlo Risk</button>
                    </div>

                    {/* SECTION 2: MODEL EVALUATION */}
                    {(deepSubTab === 'all' || deepSubTab === 'backtest') && (
                      <div id="sec-eval" style={{ marginTop: '1.5rem' }}>
                        <div className="glass-panel" style={{ padding: '1.5rem' }}>
                        {(() => {
                          const dates = backtestMetrics?.dates || [];
                          const startDate = dates.length > 0 ? dates[0].split(' ')[0] : '';
                          const endDate = dates.length > 0 ? dates[dates.length - 1].split(' ')[0] : '';
                          const dateRangeText = startDate && endDate ? `(${startDate} to ${endDate})` : '';
                          return (
                            <h3 style={{ margin: '0 0 1rem 0', fontSize: '1.1rem', borderBottom: '1px solid var(--glass-border)', paddingBottom: '0.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
                              <span>📈 Walk-Forward Backtest Performance <InfoBadge topic="Walk-Forward Testing" text="Out-of-sample simulation mimicking real-world forward propagation." anchor="#rl-basics" onNavigate={handleNavigateSchool} /></span>
                              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                                {dateRangeText && <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontWeight: 'normal' }}>{startDate} to {endDate}</span>}
                                {backtestMetrics && (
                                  <button
                                    onClick={() => fetchAnalytics(selectedAsset.label, selectedRegime, true)}
                                    disabled={loadingAnalytics}
                                    style={{
                                      background: 'rgba(255, 255, 255, 0.08)',
                                      border: '1px solid var(--glass-border)',
                                      color: '#FFFFFF',
                                      borderRadius: '6px',
                                      padding: '0.3rem 0.75rem',
                                      fontSize: '0.78rem',
                                      cursor: 'pointer',
                                      display: 'flex',
                                      alignItems: 'center',
                                      gap: '0.35rem'
                                    }}
                                  >
                                    🔄 Re-run Backtest
                                  </button>
                                )}
                              </div>
                            </h3>
                          );
                        })()}
                        {loadingBacktest ? (
                          <div style={{ padding: '3.5rem 2rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
                            <div className="skeleton" style={{ width: '100%', height: '180px', marginBottom: '1.25rem', opacity: 0.25 }}></div>
                            <div style={{ fontSize: '1rem', color: '#FFFFFF', fontWeight: 600 }}>
                              Running historical backtest simulation on {selectedAsset.label}...
                            </div>
                            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.35rem' }}>
                              Evaluating 1,000+ sequential 30-min trading decisions with 0.10% commission & 0.05% slippage friction.
                            </div>
                          </div>
                        ) : !backtestMetrics ? (
                          <div style={{ padding: '3rem 2rem', textAlign: 'center', background: 'rgba(255, 255, 255, 0.02)', borderRadius: '10px', border: '1px dashed var(--glass-border)' }}>
                            <div style={{ fontSize: '1.05rem', fontWeight: 600, color: '#FFFFFF', marginBottom: '0.4rem' }}>
                              Walk-Forward Backtest Simulation
                            </div>
                            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', maxWidth: '520px', margin: '0 auto 1.5rem auto', lineHeight: 1.5 }}>
                              Execute a comprehensive out-of-sample backtest for <strong>{selectedAsset.label}</strong> across historical 30-min bars. Evaluates Sharpe ratio, maximum drawdown, total return, and trade-by-trade equity trajectories under real market friction.
                            </p>
                            <button
                              onClick={() => fetchAnalytics(selectedAsset.label, selectedRegime, true)}
                              style={{
                                background: '#FFFFFF',
                                color: '#000000',
                                border: 'none',
                                borderRadius: '8px',
                                padding: '0.65rem 1.6rem',
                                fontSize: '0.9rem',
                                fontWeight: 700,
                                cursor: 'pointer',
                                display: 'inline-flex',
                                alignItems: 'center',
                                gap: '0.5rem',
                                boxShadow: '0 2px 12px rgba(255, 255, 255, 0.15)',
                                transition: 'all 0.18s ease'
                              }}
                            >
                              ▶ Run Backtest Simulation
                            </button>
                          </div>
                        ) : (
                          <>
                            <div style={{ marginTop: '1rem', padding: '1.25rem', background: 'rgba(0,0,0,0.1)', borderRadius: '12px', border: '1px solid var(--glass-border)' }}>
                              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.85rem', flexWrap: 'wrap', gap: '0.75rem' }}>
                                <div>
                                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flexWrap: 'wrap' }}>
                                    <h4 style={{ margin: 0, fontSize: '1rem', color: '#FFFFFF', fontWeight: 600 }}>
                                      AI Model Historical Performance Curve (% Return)
                                    </h4>
                                    <span style={{
                                      fontSize: '0.72rem',
                                      fontWeight: 700,
                                      color: backtestMetrics.roi >= 0 ? '#34D399' : '#F87171',
                                      background: backtestMetrics.roi >= 0 ? 'rgba(52, 211, 153, 0.12)' : 'rgba(248, 113, 113, 0.12)',
                                      border: `1px solid ${backtestMetrics.roi >= 0 ? 'rgba(52, 211, 153, 0.25)' : 'rgba(248, 113, 113, 0.25)'}`,
                                      padding: '0.15rem 0.55rem',
                                      borderRadius: '12px'
                                    }}>
                                      ● {backtestMetrics.roi >= 0 ? '+' : ''}{Number(backtestMetrics.roi).toFixed(2)}% Net Return
                                    </span>
                                    <span style={{
                                      fontSize: '0.72rem',
                                      color: 'var(--text-secondary)',
                                      background: 'rgba(255, 255, 255, 0.05)',
                                      border: '1px solid var(--glass-border)',
                                      padding: '0.15rem 0.55rem',
                                      borderRadius: '12px'
                                    }}>
                                      {backtestMetrics.trades} Trades
                                    </span>
                                  </div>
                                  <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)', marginTop: '0.35rem' }}>
                                    Deterministic out-of-sample backtest across historical candles ({backtestMetrics.dates && backtestMetrics.dates.length > 0 ? `${backtestMetrics.dates[0].split(' ')[0]} to ${backtestMetrics.dates[backtestMetrics.dates.length - 1].split(' ')[0]}` : 'historical period'})
                                  </div>
                                </div>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                                  {showTradeMarkers && (
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                                      <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem', fontSize: '0.72rem' }}>
                                        <span style={{ display: 'inline-block', width: '7px', height: '7px', background: '#34D399', transform: 'rotate(45deg)' }}></span> Buy
                                      </span>
                                      <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem', fontSize: '0.72rem' }}>
                                        <span style={{ display: 'inline-block', width: '7px', height: '7px', background: '#F87171', transform: 'rotate(45deg)' }}></span> Sell
                                      </span>
                                    </div>
                                  )}
                                  <button
                                    onClick={() => setShowTradeMarkers(prev => !prev)}
                                    style={{
                                      background: showTradeMarkers ? 'rgba(255, 255, 255, 0.12)' : 'rgba(255, 255, 255, 0.04)',
                                      border: '1px solid var(--glass-border)',
                                      color: showTradeMarkers ? '#FFFFFF' : 'var(--text-muted)',
                                      borderRadius: '6px',
                                      padding: '0.25rem 0.65rem',
                                      fontSize: '0.74rem',
                                      cursor: 'pointer',
                                      display: 'flex',
                                      alignItems: 'center',
                                      gap: '0.35rem',
                                      transition: 'all 0.15s ease'
                                    }}
                                  >
                                    {showTradeMarkers ? 'Hide Trade Markers' : `Show Trade Markers (${backtestMetrics.trades})`}
                                  </button>
                                </div>
                              </div>
                              {backtestMetrics.history && (() => {
                                const totalSteps = backtestMetrics.history.length;
                                const maxRenderPoints = 350;
                                const stride = Math.max(1, Math.floor(totalSteps / maxRenderPoints));

                                // Build O(1) lookup map for trade markers
                                const tradeMap = new Map<number, TradeItem>();
                                if (backtestMetrics.trade_history) {
                                  for (let k = 0; k < backtestMetrics.trade_history.length; k++) {
                                    const t = backtestMetrics.trade_history[k];
                                    tradeMap.set(t.step, t);
                                  }
                                }

                                const baseVal = backtestMetrics.history[0] || 10000;
                                let currentPeak = backtestMetrics.history[0];
                                let minReturn = Infinity;
                                let maxReturn = -Infinity;
                                let ddMin = 0;

                                const chartData = [];
                                for (let idx = 0; idx < totalSteps; idx++) {
                                  const val = backtestMetrics.history[idx];
                                  if (val > currentPeak) currentPeak = val;
                                  
                                  const returnPct = baseVal > 0 ? ((val - baseVal) / baseVal) * 100 : 0;
                                  const dd = ((val - currentPeak) / currentPeak) * 100;

                                  const trade = tradeMap.get(idx);
                                  const isBuy = trade && (trade.action === 1 || trade.action === 4);
                                  const isSell = trade && (trade.action === 2 || trade.action === 3);

                                  // Get date label if available
                                  const rawDate = backtestMetrics.dates && backtestMetrics.dates[idx] ? backtestMetrics.dates[idx] : `Step ${idx}`;
                                  const shortDate = rawDate.includes(' ') ? rawDate.split(' ')[0] : rawDate;

                                  // Downsample: include points at regular stride intervals, trade points, and endpoints
                                  if (idx % stride === 0 || trade || idx === totalSteps - 1) {
                                    if (returnPct < minReturn) minReturn = returnPct;
                                    if (returnPct > maxReturn) maxReturn = returnPct;
                                    if (dd < ddMin) ddMin = dd;

                                    chartData.push({
                                      step: idx,
                                      date: shortDate,
                                      dateFull: rawDate,
                                      returnPct: Math.round(returnPct * 100) / 100,
                                      drawdown: Math.round(dd * 100) / 100,
                                      buyMarker: isBuy ? Math.round(returnPct * 100) / 100 : null,
                                      sellMarker: isSell ? Math.round(returnPct * 100) / 100 : null
                                    });
                                  }
                                }

                                const yDomain: [number | string, number | string] = minReturn === maxReturn || !isFinite(minReturn) ? [-5, 5] : ['auto', 'auto'];
                                const ddDomain: [number | string, number | string] = ddMin === 0 ? [-1, 0] : ['auto', 0];

                                return (
                                  <>
                                    <div style={{ height: 250, width: '100%', marginTop: '0.75rem' }}>
                                      <ResponsiveContainer width="100%" height="100%">
                                        <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                                          <XAxis 
                                            dataKey="date" 
                                            stroke="rgba(255,255,255,0.2)" 
                                            fontSize={10} 
                                            interval="preserveStartEnd"
                                            minTickGap={60}
                                          />
                                          <YAxis 
                                            stroke="rgba(255,255,255,0.2)" 
                                            fontSize={10} 
                                            domain={yDomain} 
                                            tickFormatter={(v) => `${Number(v) >= 0 ? '+' : ''}${Number(v).toFixed(1)}%`} 
                                          />
                                          <RechartsTooltip 
                                            contentStyle={{ background: 'rgba(12, 12, 14, 0.96)', border: '1px solid rgba(255, 255, 255, 0.18)', borderRadius: '8px' }} 
                                            labelStyle={{ color: '#FFFFFF', fontWeight: 'bold', marginBottom: '0.2rem' }}
                                            labelFormatter={(label, items) => {
                                              if (items && items[0] && items[0].payload && items[0].payload.dateFull) {
                                                return items[0].payload.dateFull;
                                              }
                                              return label;
                                            }}
                                            formatter={(value, name) => {
                                              if (name === 'Strategy Return' && typeof value === 'number') return [`${value >= 0 ? '+' : ''}${value.toFixed(2)}%`, 'Strategy Return'];
                                              if (name === 'drawdown' && typeof value === 'number') return [`${value.toFixed(2)}%`, 'Drawdown'];
                                              return [null, null];
                                            }}
                                          />
                                           <Line 
                                             type="stepAfter" 
                                             dataKey="returnPct" 
                                             stroke={backtestMetrics.roi >= 0 ? '#34D399' : '#F87171'} 
                                             strokeWidth={2} 
                                             dot={false} 
                                             isAnimationActive={false} 
                                             name="Strategy Return" 
                                           />
                                           {showTradeMarkers && (
                                             <>
                                               <Scatter dataKey="buyMarker" fill="#34D399" shape="triangle" isAnimationActive={false} name="Buy execution" />
                                               <Scatter dataKey="sellMarker" fill="#F87171" shape="diamond" isAnimationActive={false} name="Sell execution" />
                                             </>
                                           )}
                                        </ComposedChart>
                                      </ResponsiveContainer>
                                    </div>
                                    <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Markers show executed trades, including automatic exits at the holding-time limit. A sell execution does not necessarily mean PPO chose Cash or Short.</p>
                                    <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '1.5rem', marginBottom: '0.5rem' }}>Peak-to-Trough Drawdown</div>
                                    <div style={{ height: 100, width: '100%' }}>
                                      <ResponsiveContainer width="100%" height="100%">
                                        <AreaChart data={chartData} margin={{ top: 0, right: 10, left: 0, bottom: 0 }}>
                                          <XAxis dataKey="date" hide />
                                          <YAxis hide domain={ddDomain} />
                                          <RechartsTooltip 
                                            contentStyle={{ background: 'rgba(12, 12, 14, 0.96)', border: '1px solid rgba(248, 113, 113, 0.35)', borderRadius: '8px' }} 
                                            labelFormatter={(label, items) => {
                                              if (items && items[0] && items[0].payload && items[0].payload.dateFull) {
                                                return items[0].payload.dateFull;
                                              }
                                              return label;
                                            }}
                                            formatter={(value) => [`${typeof value === 'number' ? value.toFixed(2) : value}%`, 'Drawdown']}
                                          />
                                          <Area type="stepAfter" dataKey="drawdown" stroke="#F87171" fill="#F87171" fillOpacity={0.25} isAnimationActive={false} />
                                        </AreaChart>
                                      </ResponsiveContainer>
                                    </div>
                                  </>
                                );
                              })()}
                            </div>
                            
                            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '1rem', fontStyle: 'italic' }}>
                              * Backtest executed deterministically across {backtestMetrics.dates && backtestMetrics.dates.length > 0 ? `${backtestMetrics.dates[0].split(' ')[0]} to ${backtestMetrics.dates[backtestMetrics.dates.length - 1].split(' ')[0]}` : 'the validation dataset'} with 0.1% commission and 0.05% slippage friction parameters.
                            </p>
                          </>
                        )}
                      </div>
                    </div>
                    )}

                    {/* SECTION 3: MARKET CLIMATE & REGIME ANALYSIS */}
                    {(deepSubTab === 'all' || deepSubTab === 'regime') && (
                      <div id="sec-regime" style={{ scrollMarginTop: '120px', marginTop: '2rem' }}>
                      <div className="glass-panel" style={{ padding: '1.5rem' }}>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid var(--glass-border)', paddingBottom: '0.75rem', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '0.5rem' }}>
                          <h3 style={{ margin: 0, fontSize: '1.15rem', color: '#FFFFFF', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            🌦️ Market Climate & Environment Detection
                          </h3>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
                            <span style={{ fontSize: '0.72rem', background: 'rgba(96, 165, 250, 0.12)', color: '#60A5FA', border: '1px solid rgba(96, 165, 250, 0.25)', padding: '0.2rem 0.6rem', borderRadius: '6px', fontWeight: 600 }}>
                              Unsupervised HMM (3-State Gaussian)
                            </span>
                            <span style={{ fontSize: '0.72rem', background: 'rgba(52, 211, 153, 0.12)', color: '#34D399', border: '1px solid rgba(52, 211, 153, 0.25)', padding: '0.2rem 0.6rem', borderRadius: '6px', fontWeight: 600 }}>
                              Online State Inference
                            </span>
                          </div>
                        </div>

                        {(() => {
                          const rawRegime = (selectedAsset.regime || '').toLowerCase();
                          const isBull = rawRegime.includes('bull') || rawRegime.includes('calm');
                          const isBear = rawRegime.includes('bear') || rawRegime.includes('volatile') || rawRegime.includes('crash');
                          const isSideways = !isBull && !isBear;

                          const regimes = [
                            {
                              id: 'bull',
                              name: '🌤️ Bullish Expansion',
                              badge: 'Market context',
                              desc: 'Sustained upward price trend with contained volatility. This describes market conditions, not a model-selected allocation.',
                              color: '#34D399',
                              bg: 'rgba(52, 211, 153, 0.08)',
                              border: isBull ? '#34D399' : 'rgba(255, 255, 255, 0.08)',
                              active: isBull,
                              allocation: 'Regime context only; see policy target'
                            },
                            {
                              id: 'sideways',
                              name: '🌪️ Sideways Consolidation',
                              badge: 'Market context',
                              desc: 'Rangebound chop with conflicting signals. Trading can incur fees and whipsaw losses; the policy target is shown separately.',
                              color: '#FBBF24',
                              bg: 'rgba(251, 191, 36, 0.08)',
                              border: isSideways ? '#FBBF24' : 'rgba(255, 255, 255, 0.08)',
                              active: isSideways,
                              allocation: 'Regime context only; see policy target'
                            },
                            {
                              id: 'bear',
                              name: '⚡ Bearish Downturn',
                              badge: 'Market context',
                              desc: 'High downside volatility and distribution. A bearish regime does not imply a cash position; consult the separate policy target.',
                              color: '#F87171',
                              bg: 'rgba(248, 113, 113, 0.08)',
                              border: isBear ? '#F87171' : 'rgba(255, 255, 255, 0.08)',
                              active: isBear,
                              allocation: 'Regime context only; see policy target'
                            }
                          ];

                          return (
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                              {/* 3-State Phase Meter Grid */}
                              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1rem' }}>
                                {regimes.map(r => (
                                  <div
                                    key={r.id}
                                    style={{
                                      background: r.active ? r.bg : 'rgba(255, 255, 255, 0.02)',
                                      border: `1.5px solid ${r.border}`,
                                      borderRadius: '10px',
                                      padding: '1.1rem',
                                      boxShadow: r.active ? `0 0 20px ${r.color}25` : 'none',
                                      transition: 'all 0.2s ease',
                                      position: 'relative'
                                    }}
                                  >
                                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.6rem' }}>
                                      <div style={{ fontSize: '0.95rem', fontWeight: 700, color: '#FFFFFF' }}>{r.name}</div>
                                      {r.active && (
                                        <span style={{ fontSize: '0.68rem', fontWeight: 700, background: r.color, color: '#09090B', padding: '0.15rem 0.5rem', borderRadius: '12px' }}>
                                          ACTIVE
                                        </span>
                                      )}
                                    </div>
                                    <div style={{ fontSize: '0.72rem', color: r.color, fontWeight: 600, marginBottom: '0.5rem' }}>
                                      {r.badge}
                                    </div>
                                    <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: 1.45, margin: '0 0 0.85rem 0' }}>
                                      {r.desc}
                                    </p>
                                    <div style={{ borderTop: '1px solid rgba(255,255,255,0.06)', paddingTop: '0.6rem', display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem' }}>
                                      <span style={{ color: 'var(--text-muted)' }}>Target Allocation:</span>
                                      <span style={{ fontWeight: 700, color: r.active ? r.color : 'var(--text-secondary)' }}>{r.allocation}</span>
                                    </div>
                                  </div>
                                ))}
                              </div>

                              {/* Model Snapshot Synthesis Deep-Dive */}
                              <div style={{ background: 'rgba(0, 0, 0, 0.35)', borderRadius: '10px', padding: '1.25rem', border: '1px solid rgba(255, 255, 255, 0.08)' }}>
                                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem', flexWrap: 'wrap', gap: '0.5rem' }}>
                                  <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                                    CURRENTLY CLASSIFIED CLIMATE FOR {selectedAsset.label}
                                  </div>
                                  <div style={{ fontSize: '0.85rem', fontWeight: 700, color: isBull ? '#34D399' : isBear ? '#F87171' : '#FBBF24' }}>
                                    {getPlainRegime(selectedAsset.regime)}
                                  </div>
                                </div>
                                <p style={{ fontSize: '0.85rem', color: '#E4E4E7', lineHeight: 1.5, margin: 0 }}>
                                  {selectedAsset.signal === 'BUY'
                                    ? `For ${selectedAsset.label}, the policy targets Long under the currently classified ${getPlainRegime(selectedAsset.regime).toLowerCase()} regime.`
                                    : selectedAsset.signal === 'SELL'
                                    ? `For ${selectedAsset.label}, the policy targets Short under the currently classified ${getPlainRegime(selectedAsset.regime).toLowerCase()} regime.`
                                    : `For ${selectedAsset.label}, the policy targets Cash under the currently classified ${getPlainRegime(selectedAsset.regime).toLowerCase()} regime.`}
                                </p>
                                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.85rem', fontStyle: 'italic', borderTop: '1px solid rgba(255,255,255,0.06)', paddingTop: '0.6rem' }}>
                                  * The regime is a model classification based on the loaded return and volume history (Hamilton, 1989; Rabiner, 1989).
                                </div>
                              </div>
                            </div>
                          );
                        })()}
                      </div>
                    </div>
                    )}

                    {/* SECTION 4: SHAP EXPLAINABILITY */}
                    {(deepSubTab === 'all' || deepSubTab === 'shap') && (
                      <div id="sec-shap" style={{ scrollMarginTop: '120px', marginTop: '2rem' }}>
                      <div className="glass-panel" style={{ padding: '1.5rem' }}>
                        <h3 style={{ margin: '0 0 1rem 0', fontSize: '1.1rem', borderBottom: '1px solid var(--glass-border)', paddingBottom: '0.5rem' }}>🔍 SHAP Feature Importance</h3>
                        {loadingShap ? (
                          <div style={{ padding: '2.5rem 1.5rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
                            <div className="typing-indicator" style={{ margin: 'auto' }}><span></span><span></span><span></span></div>
                            <div style={{ marginTop: '1rem', color: '#FFFFFF', fontSize: '0.9rem' }}>Computing SHAP attribution values for the latest decision weights...</div>
                          </div>
                        ) : !shapData ? (
                          <div style={{ padding: '2.5rem 1.5rem', textAlign: 'center', color: 'var(--text-secondary)', background: 'rgba(255, 255, 255, 0.01)', borderRadius: '8px', border: '1px dashed var(--glass-border)' }}>
                            <p style={{ margin: '0 auto 1.25rem auto', fontSize: '0.88rem', maxWidth: '480px' }}>
                              SHAP feature importance calculates which factors (RSI, GARCH volatility, FinBERT sentiment, MACD) most heavily influenced the latest PPO policy decision for <strong>{selectedAsset.label}</strong>.
                            </p>
                            <button
                              onClick={() => fetchAnalytics(selectedAsset.label, selectedRegime, true)}
                              style={{
                                background: 'rgba(255, 255, 255, 0.08)',
                                color: '#FFFFFF',
                                border: '1px solid var(--glass-border)',
                                borderRadius: '6px',
                                padding: '0.45rem 1.2rem',
                                fontSize: '0.82rem',
                                cursor: 'pointer',
                                fontWeight: 500
                              }}
                            >
                              🔍 Compute Feature Attribution
                            </button>
                          </div>
                        ) : (
                          <>
                            <div style={{ marginTop: '1rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                              {(() => {
                                const totalAbs = shapData.shap.reduce((acc: number, x: ShapItem) => acc + Math.abs(x.value), 0) || 1e-6;

                                const validItems = shapData.shap
                                  .map((item: ShapItem) => {
                                    const pct = item.importance_pct != null
                                      ? item.importance_pct
                                      : Math.round((Math.abs(item.value) / totalAbs) * 1000) / 10;
                                    return { ...item, pct };
                                  })
                                  .filter(item => Math.abs(item.pct) >= 0.5);

                                const displayItems = validItems.length > 0 ? validItems.slice(0, 6) : shapData.shap.slice(0, 3).map(x => ({ ...x, pct: 0 }));

                                return displayItems.map((item, idx: number) => {
                                  const widthPct = Math.min(100, Math.max(6, item.pct));
                                  const isPositive = item.value >= 0;
                                  const barColor = isPositive ? '#34D399' : '#F87171';
                                  const displayName = SHAP_FEATURE_CONFIG[item.feature]?.label || item.feature;

                                  return (
                                    <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                                      <div style={{ width: '190px', fontSize: '0.84rem', color: '#E4E4E7', textAlign: 'right', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', fontWeight: 500 }} title={displayName}>
                                        {displayName}
                                      </div>
                                      <div style={{ flex: 1, background: 'rgba(255,255,255,0.06)', height: '14px', borderRadius: '7px', overflow: 'hidden' }}>
                                        <div style={{ background: barColor, width: `${widthPct}%`, height: '100%', borderRadius: '7px', transition: 'width 0.4s ease' }}></div>
                                      </div>
                                      <div style={{ width: '85px', fontSize: '0.8rem', fontWeight: 700, color: barColor, textAlign: 'right' }} title={`Raw attribution: ${item.value >= 0 ? '+' : ''}${item.value.toExponential(3)}`}>
                                        {isPositive ? '+' : '-'}{item.pct.toFixed(1)}% Impact
                                      </div>
                                    </div>
                                  );
                                });
                              })()}
                            </div>
                            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '1.25rem', lineHeight: 1.4 }}>
                              * Features are ranked by <strong>normalized relative decision impact</strong> (|SHAP| / Σ|SHAP|). 
                              <span style={{ color: '#34D399', fontWeight: 600 }}> Green (+)</span> indicates the indicator reinforced the chosen action; 
                              <span style={{ color: '#F87171', fontWeight: 600 }}> Red (-)</span> indicates the indicator acted as a headwind against the chosen action.
                            </p>
                          </>
                        )}
                      </div>
                    </div>
                    )}

                    {/* SECTION 5: MULTI-YEAR WALK-FORWARD VALIDATION */}
                    {(deepSubTab === 'all' || deepSubTab === 'wfv') && (
                      <div id="sec-wfv" style={{ scrollMarginTop: '120px', marginTop: '2rem' }}>
                        <div className="glass-panel" style={{ padding: '1.5rem' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem', borderBottom: '1px solid var(--glass-border)', paddingBottom: '0.75rem', marginBottom: '1.25rem' }}>
                            <div>
                              <h3 style={{ margin: '0 0 0.35rem 0', fontSize: '1.15rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                📊 Multi-Year Stress Test Across Market Eras (Walk-Forward Validation)
                                <InfoBadge topic="Walk-Forward CV" text="Evaluates agent across 5 sequential out-of-sample temporal windows (2020–2026) to test real-world generalization across bull, bear, and choppy regimes without lookahead bias." anchor="#walk-forward" onNavigate={handleNavigateSchool} />
                              </h3>
                              <p style={{ margin: 0, fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                                Out-of-sample evaluation across sequential historical market periods (2021–2026) to assess policy performance on unseen data.
                              </p>
                            </div>
                            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                              <span style={{ fontSize: '0.75rem', padding: '0.25rem 0.6rem', borderRadius: '4px', background: 'rgba(251, 191, 36, 0.1)', color: '#fbbf24', border: '1px solid rgba(251, 191, 36, 0.25)', fontWeight: 600 }}>
                                Statistical Validation: Empirical DSR Not Verified
                              </span>
                              <span style={{ fontSize: '0.75rem', padding: '0.25rem 0.6rem', borderRadius: '4px', background: 'rgba(56, 189, 248, 0.1)', color: '#38bdf8', border: '1px solid rgba(56, 189, 248, 0.25)', fontWeight: 600 }}>
                                5 Historical Cycles Tested
                              </span>
                            </div>
                          </div>
                          
                          {loadingWFV ? (
                            <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
                              <div className="typing-indicator" style={{ margin: 'auto' }}><span></span><span></span><span></span></div>
                              <div style={{ marginTop: '1rem' }}>Loading out-of-sample walk-forward validation results...</div>
                            </div>
                          ) : wfvData && wfvData.length > 0 ? (
                            <>
                              {(() => {
                                const sharpes = wfvData.map(r => Number(r['Out-of-Sample Sharpe']) || 0);
                                const avgOOF = (sharpes.reduce((a, b) => a + b, 0) / Math.max(1, sharpes.length)).toFixed(2);
                                return (
                                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginBottom: '1.25rem' }}>
                                    <div className="glass-panel" style={{ padding: '1rem', textAlign: 'center' }}>
                                      <div style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>Average Out-of-Sample Efficiency</div>
                                      <div style={{ fontSize: '1.4rem', fontWeight: 'bold', color: '#FFFFFF', marginTop: '0.2rem' }}>{avgOOF}</div>
                                      <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>Sharpe ratio across 5 unseen cycles</div>
                                    </div>
                                    <div className="glass-panel" style={{ padding: '1rem', textAlign: 'center' }}>
                                      <div style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>Statistical Validation Framework</div>
                                      <div style={{ fontSize: '1.1rem', fontWeight: 'bold', color: '#fbbf24', marginTop: '0.2rem' }}>Empirical DSR Not Verified</div>
                                      <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>Bailey &amp; López de Prado (2014) • Illustrative calculation only</div>
                                    </div>
                                    <div className="glass-panel" style={{ padding: '1rem', textAlign: 'center' }}>
                                      <div style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>Validation Protocol</div>
                                      <div style={{ fontSize: '1.2rem', fontWeight: 'bold', marginTop: '0.2rem' }}>5 Sequential Eras</div>
                                      <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>Zero lookahead / future leakage</div>
                                    </div>
                                  </div>
                                );
                              })()}

                              {/* Layman Translation Callout Banner */}
                              <div style={{
                                background: 'linear-gradient(135deg, rgba(56, 189, 248, 0.08) 0%, rgba(74, 222, 128, 0.04) 100%)',
                                border: '1px solid rgba(56, 189, 248, 0.25)',
                                borderRadius: '10px',
                                padding: '1rem 1.25rem',
                                marginBottom: '1.25rem'
                              }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem' }}>
                                  <span style={{ fontSize: '1.1rem' }}>💡</span>
                                  <span style={{ fontWeight: 700, color: '#FFFFFF', fontSize: '0.9rem' }}>
                                    How to read this table & Why the AI held 100% cash in 2022–2024 (Cycles 3 & 4):
                                  </span>
                                </div>
                                <p style={{ margin: 0, fontSize: '0.82rem', color: '#E4E4E7', lineHeight: 1.5 }}>
                                  In <strong>Cycle 2 (Crypto Winter)</strong>, the market crashed <strong style={{ color: '#f87171' }}>-20.6%</strong>, but the AI achieved <strong style={{ color: '#4ade80' }}>+44.5%</strong> (+65.2% outperformance) by taking defensive short positions. Because the model learned high penalties for downside risk, when the market entered the turbulent 2022–2024 recovery (Cycles 3 & 4), the AI chose <strong>100% capital preservation (0% drawdown)</strong> rather than risking whipsaw losses. In Cycle 5 (2025–2026), when the market crashed <strong style={{ color: '#f87171' }}>-40.9%</strong>, the AI again protected capital and generated <strong style={{ color: '#4ade80' }}>+4.7%</strong>.
                                </p>
                              </div>

                              <div style={{ overflowX: 'auto', borderRadius: '8px', border: '1px solid var(--glass-border)' }}>
                                <table className="leaderboard-table" style={{ width: '100%', borderCollapse: 'collapse' }}>
                                  <thead>
                                    <tr style={{ background: 'rgba(255, 255, 255, 0.03)', textAlign: 'left' }}>
                                      <th style={{ padding: '0.85rem 1rem' }}>Market Era / Cycle</th>
                                      <th style={{ padding: '0.85rem 1rem' }}>Regime Dynamics</th>
                                      <th style={{ padding: '0.85rem 1rem', textAlign: 'right' }}>AI Return (Unseen Test)</th>
                                      <th style={{ padding: '0.85rem 1rem', textAlign: 'right' }}>Market Benchmark (B&H)</th>
                                      <th style={{ padding: '0.85rem 1rem', textAlign: 'right' }}>AI Advantage (Alpha α)</th>
                                      <th style={{ padding: '0.85rem 1rem', textAlign: 'center' }}>Efficiency (Sharpe)</th>
                                      <th style={{ padding: '0.85rem 1rem', textAlign: 'right' }}>Deepest Dip (DD)</th>
                                      <th style={{ padding: '0.85rem 1rem', textAlign: 'center' }}>Trading Activity</th>
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {wfvData.map((row: WalkForwardRow, idx: number) => {
                                      const eraInfo = WFV_ERA_MAP[row.Fold] || {
                                        era: `Cycle ${row.Fold}`,
                                        regime: 'Out-of-Sample Window',
                                        regimeColor: 'var(--text-secondary)',
                                        description: 'Sequential Testing Period'
                                      };
                                      const roi = Number(row['Out-of-Sample ROI (%)']) || 0;
                                      const market = Number(row['Market B&H (%)']) || 0;
                                      const alpha = roi - market;
                                      const sharpe = Number(row['Out-of-Sample Sharpe']) || 0;
                                      const maxDd = Number(row['Max DD (%)']) || 0;
                                      const trades = Number(row.Trades) || 0;

                                      return (
                                        <tr key={idx} className="leaderboard-row" style={{ borderTop: '1px solid rgba(255, 255, 255, 0.05)' }}>
                                          <td style={{ padding: '0.85rem 1rem' }}>
                                            <div style={{ fontWeight: 600, color: '#FFFFFF' }}>{eraInfo.era}</div>
                                            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Cycle {row.Fold} • Unseen Test Epoch</div>
                                          </td>
                                          <td style={{ padding: '0.85rem 1rem' }}>
                                            <span style={{ 
                                              display: 'inline-block', 
                                              fontSize: '0.78rem', 
                                              padding: '0.2rem 0.55rem', 
                                              borderRadius: '4px', 
                                              fontWeight: 600, 
                                              background: `${eraInfo.regimeColor}18`, 
                                              color: eraInfo.regimeColor, 
                                              border: `1px solid ${eraInfo.regimeColor}40` 
                                            }}>
                                              {eraInfo.regime}
                                            </span>
                                            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>{eraInfo.description}</div>
                                          </td>
                                          <td style={{ padding: '0.85rem 1rem', textAlign: 'right', fontWeight: 600 }}>
                                            <span style={{ 
                                              color: roi > 0 ? '#4ade80' : roi === 0 ? 'var(--text-secondary)' : '#f87171',
                                              fontSize: '0.95rem'
                                            }}>
                                              {roi > 0 ? `+${roi.toFixed(1)}%` : `${roi.toFixed(1)}%`}
                                            </span>
                                          </td>
                                          <td style={{ padding: '0.85rem 1rem', textAlign: 'right', fontWeight: 500 }}>
                                            <span style={{ 
                                              color: market > 0 ? '#4ade80' : market === 0 ? 'var(--text-secondary)' : '#f87171',
                                              fontSize: '0.95rem'
                                            }}>
                                              {market > 0 ? `+${market.toFixed(1)}%` : `${market.toFixed(1)}%`}
                                            </span>
                                          </td>
                                          <td style={{ padding: '0.85rem 1rem', textAlign: 'right' }}>
                                            {trades === 0 ? (
                                              <span style={{ fontSize: '0.75rem', padding: '0.2rem 0.5rem', borderRadius: '4px', background: 'rgba(251, 191, 36, 0.1)', color: '#fbbf24', border: '1px solid rgba(251, 191, 36, 0.25)', fontWeight: 600 }}>
                                                Preserved Cash (0% Dip)
                                              </span>
                                            ) : (
                                              <span style={{ 
                                                fontWeight: 700, 
                                                color: alpha >= 0 ? '#4ade80' : '#f87171',
                                                fontSize: '0.95rem'
                                              }}>
                                                {alpha >= 0 ? `+${alpha.toFixed(1)}%` : `${alpha.toFixed(1)}%`}
                                              </span>
                                            )}
                                          </td>
                                          <td style={{ padding: '0.85rem 1rem', textAlign: 'center', fontWeight: 600, color: sharpe > 1.0 ? '#4ade80' : sharpe > 0 ? '#FFFFFF' : 'var(--text-secondary)' }}>
                                            {sharpe.toFixed(2)}
                                          </td>
                                          <td style={{ padding: '0.85rem 1rem', textAlign: 'right', color: maxDd === 0 ? 'var(--text-secondary)' : '#f87171', fontWeight: 500 }}>
                                            {maxDd === 0 ? '0.0%' : `${maxDd.toFixed(1)}%`}
                                          </td>
                                          <td style={{ padding: '0.85rem 1rem', textAlign: 'center' }}>
                                            {trades > 0 ? (
                                              <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{trades} trades</span>
                                            ) : (
                                              <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', fontStyle: 'italic' }}>0 trades (100% Cash Defense)</span>
                                            )}
                                          </td>
                                        </tr>
                                      );
                                    })}
                                  </tbody>
                                </table>
                              </div>

                              {/* ACADEMIC EVALUATION & REGIME SENSITIVITY CALLOUT BOX */}
                              <div style={{ marginTop: '1.5rem', background: 'rgba(255, 255, 255, 0.02)', border: '1px solid var(--glass-border)', borderRadius: '12px', padding: '1.4rem' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '1rem' }}>
                                  <span style={{ fontSize: '1.25rem' }}>🎓</span>
                                  <div>
                                    <h4 style={{ margin: 0, fontSize: '0.98rem', fontWeight: 600, color: '#FFFFFF' }}>
                                      Academic Evaluation & Model Limitation Analysis (Capstone Defense)
                                    </h4>
                                    <p style={{ margin: 0, fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                                      Critical analysis of behavioral asymmetry, loss-function trade-offs, and empirical findings across market cycles.
                                    </p>
                                  </div>
                                </div>

                                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.2rem' }}>
                                  {/* Pillar 1: Downside Capital Protection */}
                                  <div style={{ background: 'rgba(0, 0, 0, 0.25)', padding: '1.1rem', borderRadius: '8px', border: '1px solid rgba(74, 222, 128, 0.25)' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                                      <span style={{ color: '#4ade80', fontWeight: 'bold', fontSize: '0.9rem' }}>🛡️ Asymmetric Downside Protection (Bear Alpha)</span>
                                    </div>
                                    <p style={{ margin: 0, fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                                      In Fold 2 (<strong style={{ color: '#f87171' }}>Market -20.7%</strong>) and Fold 5 (<strong style={{ color: '#f87171' }}>Market -40.9%</strong>), the policy generated <strong style={{ color: '#4ade80' }}>+44.5%</strong> and <strong style={{ color: '#4ade80' }}>+4.7%</strong> ROI (+65.2% and +45.6% excess alpha). The deep RL agent dynamically learned to de-risk into cash and take defensive short exposures, recording positive out-of-sample returns during those historical crash periods.
                                    </p>
                                  </div>

                                  {/* Pillar 2: Critical Reflection: Folds 3 & 4 Zero Trades */}
                                  <div style={{ background: 'rgba(0, 0, 0, 0.25)', padding: '1.1rem', borderRadius: '8px', border: '1px solid rgba(251, 191, 36, 0.25)' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                                      <span style={{ color: '#fbbf24', fontWeight: 'bold', fontSize: '0.9rem' }}>⚖️ Critical Reflection: Bull Market Cash Bias</span>
                                    </div>
                                    <p style={{ margin: 0, fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                                      In Folds 3 & 4 (<strong style={{ color: '#4ade80' }}>Market +123.8% and +60.6%</strong>), the model executed <strong>0 trades</strong> (holding 100% cash). The PPO loss incorporates explicit penalties on downside variance and turnover friction (<code style={{ fontSize: '0.74rem', background: 'rgba(255,255,255,0.06)', padding: '0.1rem 0.3rem', borderRadius: '3px' }}>-λ·DD</code>). In speculative parabolic expansions without established multi-factor consensus, the agent prioritized capital preservation over unhedged risk, trading upside participation (Type II opportunity cost) for zero drawdown.
                                    </p>
                                  </div>

                                  {/* Pillar 3: Anti-Overfitting & Walk-Forward Integrity */}
                                  <div style={{ background: 'rgba(0, 0, 0, 0.25)', padding: '1.1rem', borderRadius: '8px', border: '1px solid rgba(56, 189, 248, 0.25)' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                                      <span style={{ color: '#38bdf8', fontWeight: 'bold', fontSize: '0.9rem' }}>📐 López de Prado Anti-Overfitting Evaluation</span>
                                    </div>
                                    <p style={{ margin: 0, fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                                      Evaluated under Marcos López de Prado's Deflated Sharpe Ratio framework across 5 sequential testing blocks. Note: Empirical DSR remains unverified; out-of-sample results reflect historical walk-forward folds without mathematical proof of future optimality.
                                    </p>
                                  </div>
                                </div>

                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem', marginTop: '1.2rem', paddingTop: '0.75rem', borderTop: '1px solid rgba(255, 255, 255, 0.05)', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                                  <span>Authoritative Citation: David H. Bailey & Marcos López de Prado (2014), <em>The Deflated Sharpe Ratio</em>, SSRN ID 2460551.</span>
                                  <span>Methodology: Walk-Forward Analysis (WFA) with Expanding In-Sample Windows</span>
                                </div>
                              </div>
                            </>
                          ) : (
                            <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
                              <p>No walk-forward validation data loaded.</p>
                              <button className="chat-suggested-btn" onClick={fetchWalkForward} style={{ width: 'auto', margin: '0 auto' }}>Fetch WFV Data</button>
                            </div>
                          )}
                        </div>
                      </div>
                    )}

                    {/* SECTION 6: MONTE CARLO SIMULATION */}
                    {(deepSubTab === 'all' || deepSubTab === 'montecarlo') && (
                      <div id="sec-montecarlo" style={{ scrollMarginTop: '120px', marginTop: '2rem' }}>
                        <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--glass-border)', paddingBottom: '0.5rem' }}>
                            <h3 style={{ margin: 0, fontSize: '1.1rem' }}>🎲 Monte Carlo Risk Simulation</h3>
                            <p style={{ margin: 0, fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Predict potential equity curves under synthetic price paths</p>
                          </div>

                        {/* CONFIGURATION CONTROLS */}
                        <div className="glass-panel" style={{ padding: '1.2rem', display: 'grid', gridTemplateColumns: '1.5fr 1.5fr 1fr 1fr', gap: '1.5rem', alignItems: 'center', background: 'rgba(0,0,0,0.1)' }}>
                          <div>
                            <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
                              Simulations: <span style={{ color: 'var(--text-primary)', fontWeight: 'bold' }}>{mcSimulations}</span>
                            </label>
                            <input 
                              type="range" 
                              min="10" 
                              max="100" 
                              step="10" 
                              style={{ width: '100%', accentColor: '#FFFFFF' }}
                              value={mcSimulations} 
                              onChange={e => setMcSimulations(Number(e.target.value))} 
                            />
                          </div>

                          <div>
                            <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
                              Noise Std Dev: <span style={{ color: 'var(--text-primary)', fontWeight: 'bold' }}>{mcNoise.toFixed(4)}</span>
                            </label>
                            <input 
                              type="range" 
                              min="0.001" 
                              max="0.010" 
                              step="0.001" 
                              style={{ width: '100%', accentColor: '#FFFFFF' }}
                              value={mcNoise} 
                              onChange={e => setMcNoise(Number(e.target.value))} 
                            />
                          </div>

                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <input 
                              type="checkbox" 
                              id="mcSwanCheckbox"
                              style={{ width: '16px', height: '16px', cursor: 'pointer', accentColor: '#FFFFFF' }}
                              checked={mcSwan} 
                              onChange={e => setMcSwan(e.target.checked)} 
                            />
                            <label htmlFor="mcSwanCheckbox" style={{ fontSize: '0.85rem', cursor: 'pointer', userSelect: 'none' }}>Black Swan Crash</label>
                          </div>

                          <button 
                            className="chat-suggested-btn" 
                            style={{ margin: 0, textAlign: 'center', background: '#FFFFFF', color: '#000000', border: '1px solid #FFFFFF', borderRadius: '8px', padding: '0.6rem 1rem', cursor: 'pointer', fontWeight: 'bold', boxShadow: '0 2px 12px rgba(255, 255, 255, 0.15)' }}
                            onClick={() => selectedAsset && fetchMonteCarlo(selectedAsset.label, mcSimulations, mcNoise, mcSwan)}
                            disabled={loadingMC}
                          >
                            {loadingMC ? "Simulating..." : "Run Simulation"}
                          </button>
                        </div>

                        {/* STATUS LOADER / METRICS & CHART */}
                        {loadingMC ? (
                          <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
                            <div className="typing-indicator" style={{ margin: 'auto' }}><span></span><span></span><span></span></div>
                            <div style={{ marginTop: '1rem', color: '#FFFFFF', fontSize: '0.95rem' }}>Running Monte Carlo projection paths on {selectedAsset.label}...</div>
                          </div>
                        ) : !mcData ? (
                          <div style={{ padding: '2.5rem 1.5rem', textAlign: 'center', color: 'var(--text-secondary)', background: 'rgba(255, 255, 255, 0.01)', borderRadius: '8px', border: '1px dashed var(--glass-border)' }}>
                            <p style={{ margin: 0, fontSize: '0.88rem' }}>
                              Configure parameters above and click <strong>Run Simulation</strong> to compute probabilistic geometric Brownian motion equity cones for <strong>{selectedAsset.label}</strong>.
                            </p>
                          </div>
                        ) : (
                          <>
                            {/* METRICS CARDS GRID */}
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '1rem' }}>
                              <div style={{ background: 'rgba(0,0,0,0.2)', padding: '1rem', borderRadius: '8px', textAlign: 'center', border: '1px solid var(--glass-border)' }}>
                                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Expected Return (Mean ROI)</div>
                                <div style={{ fontSize: '1.4rem', fontWeight: 'bold', marginTop: '0.25rem', color: mcData.summary.mean_roi >= 0 ? '#4ade80' : '#f87171' }}>
                                  {mcData.summary.mean_roi >= 0 ? '+' : ''}{mcData.summary.mean_roi.toFixed(2)}%
                                </div>
                                <div style={{ fontSize: '0.72rem', color: mcData.summary.mean_roi >= 0 ? '#4ade80' : '#f87171', marginTop: '0.2rem', fontWeight: 600 }}>
                                  {mcData.summary.mean_roi >= 0 ? '🟢 Net Growth' : '🔴 Net Loss'}
                                </div>
                                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>Average across simulated futures</div>
                              </div>
                              <div style={{ background: 'rgba(0,0,0,0.2)', padding: '1rem', borderRadius: '8px', textAlign: 'center', border: '1px solid var(--glass-border)' }}>
                                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Simulation Win Rate</div>
                                <div style={{ fontSize: '1.4rem', fontWeight: 'bold', marginTop: '0.25rem', color: mcData.summary.success_rate >= 50 ? '#4ade80' : '#f87171' }}>
                                  {mcData.summary.success_rate.toFixed(1)}%
                                </div>
                                <div style={{ fontSize: '0.72rem', color: mcData.summary.success_rate >= 60 ? '#4ade80' : mcData.summary.success_rate >= 50 ? '#fbbf24' : '#f87171', marginTop: '0.2rem', fontWeight: 600 }}>
                                  {mcData.summary.success_rate >= 60 ? '🟢 High Conviction' : mcData.summary.success_rate >= 50 ? '🟡 Balanced Edge' : '🔴 Skewed Risk'}
                                </div>
                                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>Futures ending in positive profit</div>
                              </div>
                              <div style={{ background: 'rgba(0,0,0,0.2)', padding: '1rem', borderRadius: '8px', textAlign: 'center', border: '1px solid var(--glass-border)' }}>
                                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Worst-Case Safety Buffer (VaR 95%)</div>
                                <div style={{ fontSize: '1.4rem', fontWeight: 'bold', marginTop: '0.25rem', color: mcData.summary.var_5th >= 0 ? '#4ade80' : '#f87171' }}>
                                  {mcData.summary.var_5th.toFixed(2)}%
                                </div>
                                <div style={{ fontSize: '0.72rem', color: mcData.summary.var_5th >= -12 ? '#4ade80' : '#fbbf24', marginTop: '0.2rem', fontWeight: 600 }}>
                                  {mcData.summary.var_5th >= -12 ? '🟢 Contained Tail Risk' : '🟡 High Volatility'}
                                </div>
                                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>95% of futures stayed above this floor</div>
                              </div>
                              <div style={{ background: 'rgba(0,0,0,0.2)', padding: '1rem', borderRadius: '8px', textAlign: 'center', border: '1px solid var(--glass-border)' }}>
                                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Mean Risk Efficiency (Sharpe)</div>
                                <div style={{ fontSize: '1.4rem', fontWeight: 'bold', marginTop: '0.25rem', color: mcData.summary.mean_sharpe >= 0 ? '#4ade80' : '#f87171' }}>
                                  {mcData.summary.mean_sharpe.toFixed(2)}
                                </div>
                                <div style={{ fontSize: '0.72rem', color: mcData.summary.mean_sharpe >= 1 ? '#4ade80' : mcData.summary.mean_sharpe >= 0 ? '#fbbf24' : '#f87171', marginTop: '0.2rem', fontWeight: 600 }}>
                                  {mcData.summary.mean_sharpe >= 1 ? '🟢 Solid Future Edge' : '🟡 Moderate Return/Risk'}
                                </div>
                                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>Return per unit of simulated swings</div>
                              </div>
                            </div>

                            {/* CUSTOM SVG EQUITY CONE CHART */}
                            <div style={{ position: 'relative', marginTop: '1rem', background: 'rgba(0,0,0,0.15)', padding: '1.2rem', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)' }}>
                              <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>Equity Projection Cone ({mcData.step_count} Simulation Steps)</div>
                              <MonteCarloConeChart mcData={mcData} />
                              
                              <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.8rem', fontStyle: 'italic', margin: 0 }}>
                                * Outer band covers 90% of outcomes (5th–95th percentile). Inner band covers 50% (25th–75th percentile). Individual lines represent random trajectory samples.
                              </p>
                            </div>
                          </>
                        )}
                      </div>
                    </div>
                    )}
                      </>
                    )}
                  </div>
                ) : (
                  <div className="glass-panel" style={{ padding: '3.5rem 2rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
                    <Cpu size={48} style={{ opacity: 0.25, marginBottom: '1rem' }} />
                    <h3 style={{ margin: '0 0 0.5rem 0', color: '#FFFFFF' }}>No Asset Selected</h3>
                    <p style={{ fontSize: '0.9rem', maxWidth: '420px', margin: '0 auto 1.5rem auto' }}>
                      Select an asset from the switcher above or return to the Trading Dashboard to inspect its deep quantitative analytics suite.
                    </p>
                    <button
                      className="chat-suggested-btn"
                      onClick={() => setCopilotCenterTab('dashboard')}
                      style={{ width: 'auto', margin: '0 auto', display: 'inline-flex', alignItems: 'center', gap: '0.5rem', padding: '0.5rem 1.25rem' }}
                    >
                      ← Return to Trading Dashboard
                    </button>
                  </div>
                )}
              </>

  );

  const renderSchoolContent = () => (
              <div className="glass-panel" style={{ padding: '2rem', maxWidth: '900px', margin: '0 auto' }}>
                {/* Back Button */}
                <div style={{ display: 'flex', alignItems: 'center', marginBottom: '1.25rem' }}>
                  <button
                    onClick={() => setCopilotCenterTab('overview')}
                    style={{
                      background: 'rgba(255, 255, 255, 0.05)',
                      border: '1px solid var(--glass-border)',
                      color: '#FFFFFF',
                      borderRadius: '8px',
                      padding: '0.45rem 1rem',
                      fontSize: '0.85rem',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.5rem',
                      transition: 'all 0.2s ease'
                    }}
                  >
                    <ArrowLeft size={16} /> Back to Terminal
                  </button>
                </div>

                <h2 style={{ marginTop: 0, display: 'flex', alignItems: 'center', gap: '0.8rem' }}>
                  <BookOpen size={26} color="#FFFFFF" />
                  Model School & Performance Benchmarks
                </h2>
                <p style={{ color: 'var(--text-secondary)', fontSize: '1.05rem', marginBottom: '2rem' }}>
                  Learn how the Adaptive AI Trading System operates, and compare PPO Transformer performance against benchmark baseline models.
                </p>

                {/* Quantitative Benchmark Comparison Card */}
                <div className="glass-panel" style={{ padding: '1.5rem', marginBottom: '2.5rem', background: 'rgba(20, 20, 24, 0.6)', borderRadius: '12px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                    <h3 style={{ margin: 0, fontSize: '1.1rem', color: '#FFFFFF' }}>🏆 Model Comparison & Out-of-Sample Benchmarks</h3>
                    <button 
                      className="chat-suggested-btn" 
                      onClick={fetchComparison} 
                      disabled={loadingComparison}
                      style={{ padding: '0.4rem 0.8rem', fontSize: '0.8rem' }}
                    >
                      {loadingComparison ? 'Refreshing...' : '🔄 Refresh Benchmarks'}
                    </button>
                  </div>
                  
                  {comparisonData ? (
                    <table className="leaderboard-table" style={{ width: '100%' }}>
                      <thead>
                        <tr>
                          <th style={{ textAlign: 'left' }}>Model Architecture</th>
                          <th style={{ textAlign: 'center' }}>Out-of-Sample Return</th>
                          <th style={{ textAlign: 'center' }}>Max Drawdown</th>
                          <th style={{ textAlign: 'center' }}>Sharpe Ratio</th>
                          <th style={{ textAlign: 'center' }}>Trades</th>
                        </tr>
                      </thead>
                      <tbody>
                        {comparisonData.map((row: ComparisonRow, idx: number) => {
                          const isPPO = row.Model.includes('PPO');
                          const roiVal = (row['ROI (%)'] !== undefined ? row['ROI (%)'] : row.ROI) ?? 0;
                          const maxDdVal = (row['Max DD (%)'] !== undefined ? row['Max DD (%)'] : row.Max_DD) ?? 0;
                          const sharpeVal = row.Sharpe !== undefined ? row.Sharpe : 0;
                          const tradesVal = row.Trades !== undefined ? row.Trades : 0;

                          return (
                            <tr key={idx} style={{ background: isPPO ? 'rgba(255, 255, 255, 0.08)' : 'transparent', fontWeight: isPPO ? 'bold' : 'normal' }}>
                              <td style={{ textAlign: 'left', color: isPPO ? '#FFFFFF' : 'var(--text-secondary)' }}>
                                {isPPO && '⭐ '} {row.Model}
                              </td>
                              <td style={{ textAlign: 'center', color: roiVal >= 0 ? '#4ade80' : '#f87171' }}>
                                {roiVal >= 0 ? '+' : ''}{roiVal}%
                              </td>
                              <td style={{ textAlign: 'center', color: '#f87171' }}>
                                {maxDdVal}%
                              </td>
                              <td style={{ textAlign: 'center', color: sharpeVal >= 0 ? '#4ade80' : '#f87171' }}>
                                {sharpeVal}
                              </td>
                              <td style={{ textAlign: 'center' }}>
                                {tradesVal}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  ) : (
                    <div style={{ textAlign: 'center', color: 'var(--text-secondary)', padding: '1rem' }}>Loading comparison benchmarks...</div>
                  )}
                </div>

                <div className="school-article" style={{ display: 'flex', flexDirection: 'column', gap: '2.5rem' }}>
                  <section id="rl-basics">
                    <h3>1. Reinforcement Learning (RL) Basics</h3>
                    <p>Unlike standard predictive models that just guess the next price, <strong>Reinforcement Learning</strong> trains an AI <em>agent</em> by letting it play a "game" against the market. The agent receives a <strong>state</strong> (market conditions), takes an <strong>action</strong> (Buy, Sell, Hold), and earns a <strong>reward</strong> (profit/loss).</p>
                    <p>Our system uses <strong>Proximal Policy Optimization (PPO)</strong>. Over millions of simulated trades, the PPO policy learns to maximize long-term risk-adjusted returns rather than just short-term gains.</p>
                  </section>

                  <section id="state-space">
                    <h3>2. How the AI Views the Market (State Space)</h3>
                    <p>The AI cannot look at charts visually like a human. Instead, it is fed a matrix of 17 key technical features stacked over the last 8 time periods (creating a temporal "memory").</p>
                    <ul>
                      <li><strong>Momentum & Trend:</strong> It tracks the Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD), and Simple Moving Averages (SMA20, SMA99).</li>
                      <li><strong>Volatility:</strong> It measures candle wicks, body ratios, and normalized volume to gauge market panic or euphoria.</li>
                      <li><strong>Portfolio State:</strong> The AI knows its current unrealized profit, how long it has been holding an asset, and whether it is flat or active.</li>
                    </ul>
                  </section>

                  <section id="hmm-regimes">
                    <h3>3. Market Regime Detection (HMM)</h3>
                    <p>Markets behave differently in bull runs versus bear crashes. We use a <strong>Hidden Markov Model (HMM)</strong> to automatically classify the market into different volatility regimes.</p>
                    <p>By understanding if the market is in a "High Volatility Bear" or "Low Volatility Bull" phase, the RL agent can dynamically adjust its risk tolerance—for example, tightening stop-losses during crashes.</p>
                  </section>

                  <section id="shap-xai">
                    <h3>4. Explainable AI (SHAP)</h3>
                    <p>Neural networks are often considered "black boxes." To solve this, our Advisor uses <strong>SHAP (SHapley Additive exPlanations)</strong>, a method from cooperative game theory.</p>
                    <p>SHAP breaks down the AI's final decision into individual feature contributions. For instance, if the AI says "BUY", SHAP can tell us that 40% of that decision came from an RSI drop, and 20% came from a volume spike. The chatbot uses this mathematical data to explain trades to you in plain English.</p>
                  </section>
                </div>
              </div>

  );

  // ─── RENDER ───────────────────────────────────────────────────────
  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', background: 'var(--bg-dark)' }}>
      {/* HEADER */}
      <header className="dashboard-header" style={{
        height: '64px',
        background: 'rgba(10, 10, 12, 0.96)',
        borderBottom: '1px solid var(--glass-border)',
        backdropFilter: 'blur(20px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 1.75rem',
        zIndex: 50,
        boxSizing: 'border-box'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div 
            onClick={() => setCopilotCenterTab('dashboard')}
            title="Return to Main Multi-Market Dashboard"
            style={{
              background: '#18181B',
              border: '1px solid rgba(255, 255, 255, 0.16)',
              padding: '0.4rem 0.85rem',
              borderRadius: '8px',
              fontWeight: 800,
              fontSize: '0.92rem',
              letterSpacing: '0.04em',
              color: '#FFFFFF',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              boxShadow: '0 2px 10px rgba(0, 0, 0, 0.5)',
              cursor: 'pointer',
              userSelect: 'none'
            }}
          >
            <Activity size={17} color="#34D399" /> TRADEPULSE AI
          </div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace', borderLeft: '1px solid rgba(255,255,255,0.1)', paddingLeft: '1rem' }}>
            SMART MARKET COPILOT
          </span>
        </div>

        {/* Right Utility Navigation: Dashboard & Model School */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>

          <button
            onClick={() => setCopilotCenterTab('dashboard')}
            style={{
              background: copilotCenterTab === 'dashboard' ? '#FFFFFF' : 'rgba(255, 255, 255, 0.04)',
              border: copilotCenterTab === 'dashboard' ? '1px solid #FFFFFF' : '1px solid var(--glass-border)',
              boxShadow: copilotCenterTab === 'dashboard' ? '0 2px 12px rgba(255, 255, 255, 0.15)' : 'none',
              color: copilotCenterTab === 'dashboard' ? '#000000' : 'var(--text-secondary)',
              fontWeight: copilotCenterTab === 'dashboard' ? 700 : 500,
              borderRadius: '8px',
              padding: '0.42rem 1.1rem',
              fontSize: '0.85rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              cursor: 'pointer',
              transition: 'all 0.18s ease'
            }}
          >
            <LayoutDashboard size={16} color={copilotCenterTab === 'dashboard' ? '#000000' : 'currentColor'} />
            <span>Dashboard</span>
          </button>

          <button
            onClick={() => setCopilotCenterTab(copilotCenterTab === 'school' ? 'dashboard' : 'school')}
            style={{
              background: copilotCenterTab === 'school' ? '#FFFFFF' : 'rgba(255, 255, 255, 0.04)',
              border: copilotCenterTab === 'school' ? '1px solid #FFFFFF' : '1px solid var(--glass-border)',
              boxShadow: copilotCenterTab === 'school' ? '0 2px 12px rgba(255, 255, 255, 0.15)' : 'none',
              color: copilotCenterTab === 'school' ? '#000000' : 'var(--text-secondary)',
              fontWeight: copilotCenterTab === 'school' ? 700 : 500,
              borderRadius: '8px',
              padding: '0.42rem 1.1rem',
              fontSize: '0.85rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              cursor: 'pointer',
              transition: 'all 0.18s ease'
            }}
          >
            <BookOpen size={16} color={copilotCenterTab === 'school' ? '#000000' : 'currentColor'} />
            <span>Model School</span>
          </button>
        </div>
      </header>

      <div className="tripane-container">
        {renderLeftRail()}
        <section className="tripane-center-canvas" ref={centerCanvasRef}>
          {copilotCenterTab === 'overview' && (
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <button
                onClick={() => setCopilotCenterTab('dashboard')}
                style={{
                  background: 'rgba(255, 255, 255, 0.05)',
                  border: '1px solid var(--glass-border)',
                  color: '#FFFFFF',
                  borderRadius: '8px',
                  padding: '0.45rem 1rem',
                  fontSize: '0.85rem',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  transition: 'all 0.2s ease'
                }}
              >
                <ArrowLeft size={16} /> Back to Dashboard
              </button>
            </div>
          )}
          {(copilotCenterTab === 'dashboard' || copilotCenterTab === 'overview') && renderSynthesisCard()}
          {copilotCenterTab === 'dashboard' && renderDashboardContent()}
          {copilotCenterTab === 'overview' && renderAnalyticsContent()}
          {copilotCenterTab === 'deep_analytics' && renderDeepAnalyticsContent()}
          {copilotCenterTab === 'school' && renderSchoolContent()}
        </section>
        {renderRightCopilotRail()}
        {isCopilotCollapsed && renderFloatingExpandBtn()}
      </div>
    </div>
  );
}

export default App;
